// This file defines the tool used for launching HIP plugins, contained in
// shared library files, as either threads or processes. Supported shared
// libraries must implement the RegisterPlugin(...) function as defined in
// plugin_interface.h.
//
// Usage: ./runner <path to a JSON config file>
// Supplying - in place of a JSON config file will cause the program to read a
// config file from stdin.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <libgen.h>
#include <errno.h>
#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "barrier_wait.h"
#include "parse_config.h"
#include "plugin_interface.h"
#include "plugin_utilities.h"

// Config files are read in chunks containing this many bytes (useful for
// reading from stdin).
#define FILE_CHUNK_SIZE (4096)

// The number of bytes used when sanitizing JSON strings for writing to the
// log. Really, the only thing that needs sanitization in here will be kernel
// names and plugin names, so this shouldn't need to be very large.
#define SANITIZE_JSON_BUFFER_SIZE (512)

// Forward declaration for the circular reference between SharedState and
// PluginState.
struct PluginState_;

// Holds data to be shared between all plugin processes or threads.
typedef struct {
  // This essentially holds all information parsed from the JSON config file.
  GlobalConfiguration *global_config;
  // The time at the start of the program, in seconds, as measured on the CPU.
  double starting_seconds;
  // This is used to force all threads or processes to wait until all are at
  // the same point in execution (e.g. after initialization).
  ProcessBarrier barrier;
  // This will be nonzero if the barrier has been successfully initialized.
  int barrier_created;
  // A list of structs holding data for each individual plugin.
  struct PluginState_ *plugins;
  // Information about the HIP device that the plugins will use.
  DeviceInformation device_info;
} SharedState;

// Holds configuration data specific to a given plugin.
typedef struct PluginState_ {
  // A reference to the PluginConfiguration for this plugin. Do not free this,
  // it will be freed when cleaning up the global_config object in SharedState.
  PluginConfiguration *config;
  PluginFunctions functions;
  // The log file for this plugin.
  FILE *output_file;
  // The handle to the plugin's shared library file, returned by dlopen.
  void *library_handle;
  // A reference to the parent shared_state structure.
  SharedState *shared_state;
  // The CPU core assigned to this plugin instance.
  int cpu_core;
} PluginState;

// The function pointer type for the registration function exported by plugin
// shared libraries.
typedef int (*RegisterPluginFunction)(PluginFunctions *functions);

// The function pointer type for the GetDeviceInformation function exported by
// the HIP utility shared library.
typedef int (*GetDeviceInfoFunction)(int device_id, DeviceInformation *info);

// Holds data about the time required to complete various phases in a single
// iteration of a plugin.
typedef struct {
  double copy_in_start;
  double copy_in_end;
  double execute_start;
  double execute_end;
  double copy_out_start;
  double copy_out_end;
} CPUTimes;

// Wraps realloc, and attempts to resize the given buffer to new_size. Returns
// 0 on error and leaves the buffer unchanged. Returns 0 on error. If buffer is
// NULL, this will allocate memory. If new_size is 0, this will free memory and
// set the buffer to NULL. Used when reading config files, including stdin.
static int SetBufferSize(void **buffer, size_t new_size) {
  void *new_pointer = NULL;
  if (new_size == 0) {
    free(*buffer);
    *buffer = NULL;
    return 1;
  }
  new_pointer = realloc(*buffer, new_size);
  if (!new_pointer) return 0;
  *buffer = new_pointer;
  return 1;
}

// Returns the TID of the calling thread.
static pid_t GetThreadID(void) {
  pid_t to_return = syscall(SYS_gettid);
  return to_return;
}

// Allocates a private shared memory buffer containing the given number of
// bytes. Can be freed by using FreeSharedBuffer. Returns NULL on error.
// Initializes the buffer to contain 0.
static void* AllocateSharedBuffer(size_t size) {
  void *to_return = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS |
    MAP_SHARED, -1, 0);
  if (to_return == MAP_FAILED) return NULL;
  memset(to_return, 0, size);
  return to_return;
}

// Frees a shared buffer returned by AllocateSharedBuffer.
static void FreeSharedBuffer(void *buffer, size_t size) {
  munmap(buffer, size);
}

// Fills in the HIP device information struct, jumping through a huge number of
// hoops to avoid loading HIP in the parent process. Returns 0 on error.
static int GetDeviceInfo(int device_id, DeviceInformation *info) {
  char exe_path[512];
  char lib_path[512];
  void *buffer = NULL;
  void *library_handle = NULL;
  GetDeviceInfoFunction info_function = NULL;
  pid_t pid = -1;
  int status;
  // Allocate a buffer in shared memory so it can be accessed by a child
  // process.
  buffer = AllocateSharedBuffer(sizeof(*info));
  if (!buffer) {
    printf("Failed allocating shared buffer for device info.\n");
  }
  memset(buffer, 0, sizeof(*info));
  pid = fork();
  if (pid < 0) {
    printf("Failed creating a child process to get GPU device info: %s\n",
      strerror(errno));
    return 0;
  }
  if (pid == 0) {
    // Load the library in the child process only! Afterwards, look up and call
    // the GetDeviceInformation function, filling in the shared buffer.
    if (readlink("/proc/self/exe", exe_path, sizeof(exe_path)) <= 0) {
      printf("Failed reading the path of the host process executable: %s\n",
        strerror(errno));
      exit(EXIT_FAILURE);
    }
    if (snprintf(lib_path, sizeof(lib_path), "%s/hip_host_utilities.so",
      dirname(exe_path)) >= sizeof(lib_path)) {
      printf("The path to hip_host_utilities.so was too long.\n");
      exit(EXIT_FAILURE);
    }
    library_handle = dlopen(lib_path, RTLD_NOW);
    if (!library_handle) {
      printf("Failed loading library %s: %s\n", lib_path, dlerror());
      exit(EXIT_FAILURE);
    }
    info_function = (GetDeviceInfoFunction) dlsym(library_handle,
      "GetDeviceInformation");
    if (!info_function) {
      printf("Failed finding the GetDeviceInformation function: %s\n",
        dlerror());
      dlclose(library_handle);
      exit(EXIT_FAILURE);
    }
    if (!info_function(device_id, (DeviceInformation *) buffer)) {
      dlclose(library_handle);
      exit(EXIT_FAILURE);
    }
    dlclose(library_handle);
    exit(EXIT_SUCCESS);
  }
  // The parent will wait for the child to finish, then copy the contents from
  // the shared buffer.
  if (wait(&status) < 0) {
    printf("failed waiting on the child process.\n");
    FreeSharedBuffer(buffer, sizeof(*info));
    return 0;
  }
  memcpy(info, buffer, sizeof(*info));
  FreeSharedBuffer(buffer, sizeof(*info));
  if (!WIFEXITED(status)) {
    printf("The child process didn't exit normally.\n");
    return 0;
  }
  if (WEXITSTATUS(status) != EXIT_SUCCESS) {
    printf("The child process exited with an error.\n");
    return 0;
  }
  return 1;
}

// Frees the PluginState objects in the list, along with the list itself.
static void FreePluginStates(PluginState *plugin_states, int plugin_count) {
  int i;
  for (i = 0; i < plugin_count; i++) {
    fclose(plugin_states[i].output_file);
    dlclose(plugin_states[i].library_handle);
  }
  memset(plugin_states, 0, plugin_count * sizeof(PluginState));
  free(plugin_states);
}

// Frees the SharedState structure, and cleans up any resources it refers to.
// This is safe to call even if the state was only partially initialized.
static void Cleanup(SharedState *state) {
  if (state->barrier_created) {
    BarrierDestroy(&(state->barrier));
  }
  if (state->plugins) {
    FreePluginStates(state->plugins, state->global_config->plugin_count);
  }
  if (state->global_config) {
    FreeGlobalConfiguration(state->global_config);
  }
  memset(state, 0, sizeof(*state));
  free(state);
}

// Takes the name of the config file and returns a pointer to a buffer
// containing its content. This will return NULL on error. On success, the
// returned buffer must be freed by the caller when no longer needed. May print
// an error message if an error occurs. If the given filename is "-" this will
// read from stdin.
static uint8_t* GetConfigFileContent(const char *filename) {
  FILE *config_file = NULL;
  uint8_t *raw_content = NULL;
  uint8_t *current_chunk_start = NULL;
  size_t total_bytes_read = 0;
  size_t last_bytes_read = 0;
  // Remember that a filename of "-" indicates to use stdin.
  if (strncmp(filename, "-", 2) == 0) {
    config_file = stdin;
  } else {
    config_file = fopen(filename, "rb");
    if (!config_file) {
      printf("Failed opening config file.\n");
      return NULL;
    }
  }
  if (!SetBufferSize((void **) (&raw_content), FILE_CHUNK_SIZE)) {
    printf("Failed allocating buffer for config file content.\n");
    if (config_file != stdin) fclose(config_file);
    return NULL;
  }
  // It would be far nicer to just allocate a chunk of memory at once, but then
  // there's no way to use stdin, since we don't know the size ahead of time.
  // Also, we need to fully buffer a file in order to parse the JSON later.
  while (1) {
    current_chunk_start = raw_content + total_bytes_read;
    memset(current_chunk_start, 0, FILE_CHUNK_SIZE);
    last_bytes_read = fread(current_chunk_start, 1, FILE_CHUNK_SIZE,
      config_file);
    // If we failed to read an entire chunk, we're either at the end of the
    // file or we encountered an error.
    if (last_bytes_read != FILE_CHUNK_SIZE) {
      if (!feof(config_file) || ferror(config_file)) {
        printf("Error reading the config.\n");
        free(raw_content);
        if (config_file != stdin) fclose(config_file);
        return NULL;
      }
      total_bytes_read += last_bytes_read;
      break;
    }
    // Allocate space for another chunk of the file to be read.
    total_bytes_read += FILE_CHUNK_SIZE;
    if (!SetBufferSize((void **) (&raw_content), total_bytes_read +
      FILE_CHUNK_SIZE)) {
      printf("Failed obtaining more memory for the config file.\n");
      free(raw_content);
      if (config_file != stdin) fclose(config_file);
      return NULL;
    }
  }
  if (config_file != stdin) fclose(config_file);
  return raw_content;
}

// This is used to cycle to the next valid CPU core in the set of available
// CPUs, since they may not be strictly in order.
static int CycleToNextCPU(int count, int current_cpu, cpu_set_t *cpu_set) {
  if (count <= 1) return current_cpu;
  // There must be at least one available CPU in the CPU set...
  while (1) {
    current_cpu = (current_cpu + 1) % count;
    if (CPU_ISSET(current_cpu, cpu_set)) return current_cpu;
  }
}

// Returns the time limit, in seconds, for the given plugin instance. Returns
// -1 if the time limit isn't set. The global time limit, if set, is overridden
// by plugin-instance-specific time limits.
static double GetTimeLimit(PluginState *state) {
  double tmp = state->config->max_time;
  if (tmp > 0) return tmp;
  tmp = state->shared_state->global_config->max_time;
  if (tmp > 0) return tmp;
  return -1;
}

// Returns the number of iterations to run the given plugin instance. Returns 0
// if the limit isn't set. If set, the plugin-instance-specific iteration limit
// overrides the global iteration limit.
static int64_t GetMaxIterations(PluginState *state) {
  int64_t tmp = state->config->max_iterations;
  if (tmp >= 0) return tmp;
  return state->shared_state->global_config->max_iterations;
}

// Allocates and initializes the plugins list in shared_state. The global
// configuration must have already been loaded for this to succeed. Returns 0
// on error.
static int CreatePluginStates(SharedState *shared_state) {
  PluginConfiguration *plugin_config = NULL;
  int i = 0;
  int cpu_count, current_cpu_core;
  PluginState *new_list = NULL;
  GlobalConfiguration *config = shared_state->global_config;
  cpu_set_t cpu_set;
  new_list = (PluginState *) malloc(config->plugin_count *
    sizeof(PluginState));
  if (!new_list) {
    printf("Failed allocating plugin states array.\n");
    return 0;
  }
  memset(new_list, 0, config->plugin_count * sizeof(PluginState));
  // This CPU count shouldn't be the number of available CPUs, but simply the
  // number at which our cyclic CPU core assignment rolls over.
  cpu_count = sysconf(_SC_NPROCESSORS_CONF);
  // Normally, start the current CPU at core 1, but there won't be a core 1 on
  // a single-CPU system.
  if (cpu_count <= 1) {
    current_cpu_core = 0;
  } else {
    current_cpu_core = 1;
  }
  CPU_ZERO(&cpu_set);
  if (sched_getaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
    printf("Failed getting CPU list.\n");
    goto ErrorCleanup;
  }
  for (i = 0; i < config->plugin_count; i++) {
    plugin_config = config->plugins + i;
    new_list[i].config = plugin_config;
    new_list[i].shared_state = shared_state;
    // Either cycle through CPUs or use the per-plugin CPU core.
    if (config->pin_cpus) {
      new_list[i].cpu_core = current_cpu_core;
      current_cpu_core = CycleToNextCPU(cpu_count, current_cpu_core, &cpu_set);
    } else {
      if ((plugin_config->cpu_core != USE_DEFAULT_CPU_CORE) && !CPU_ISSET(
        plugin_config->cpu_core, &cpu_set)) {
        printf("CPU core %d doesn't exist or isn't available.\n",
          plugin_config->cpu_core);
        goto ErrorCleanup;
      }
      new_list[i].cpu_core = plugin_config->cpu_core;
    }
    // Now try opening a log file for this plugin.
    // TODO: Make sure that no two plugins have the same log file name.
    new_list[i].output_file = fopen(plugin_config->log_name, "wb");
    if (!new_list[i].output_file) {
      printf("Failed opening log file %s: %s\n", plugin_config->log_name,
        strerror(errno));
      goto ErrorCleanup;
    }
  }
  shared_state->plugins = new_list;
  return 1;
ErrorCleanup:
  for (i = 0; i < config->plugin_count; i++) {
    if (new_list[i].output_file) fclose(new_list[i].output_file);
  }
  free(new_list);
  return 0;
}

// Takes a number of seconds to sleep. Returns 0 on error. Does nothing if the
// given amount of time is 0 or negative. Returns 1 on success.
static int SleepSeconds(double seconds) {
  if (seconds <= 0) return 1;
  if (usleep(seconds * 1e6) < 0) {
    printf("Failed sleeping %f seconds: %s\n", seconds, strerror(errno));
    return 0;
  }
  return 1;
}

// Sets the CPU affinity for the calling process or thread. Returns 0 on error
// and nonzero on success. Requires a pointer to a plugin's state to determine
// whether the caller is a process or a thread. Does nothing if the plugin
// instance's cpu_core is set to USE_DEFAULT_CPU_CORE.
static int SetCPUAffinity(PluginState *state) {
  cpu_set_t cpu_set;
  int result;
  int cpu_core = state->cpu_core;
  if (cpu_core == USE_DEFAULT_CPU_CORE) return 1;
  CPU_ZERO(&cpu_set);
  CPU_SET(cpu_core, &cpu_set);
  // Different functions are used for setting threads' and process' CPU
  // affinities.
  if (state->shared_state->global_config->use_processes) {
    result = sched_setaffinity(0, sizeof(cpu_set), &cpu_set);
  } else {
    result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
  }
  return result == 0;
}

// Copies the relevant fields out of the plugin's configuration into the given
// InitializationParameters struct. Returns 0 on error.
static int InitializeInitializationParameters(PluginState *state,
  InitializationParameters *parameters) {
  parameters->thread_count = state->config->thread_count;
  parameters->block_count = state->config->block_count;
  parameters->additional_info = state->config->additional_info;
  parameters->device_id = state->shared_state->global_config->gpu_device_id;
  memcpy(parameters->compute_unit_mask, state->config->compute_unit_mask,
    COMPUTE_UNIT_MASK_ENTRIES * sizeof(uint64_t));
  return 1;
}

// Takes a standard string and fills the output buffer with a null-terminated
// string with JSON-unsafe values properly escaped.
static void SanitizeJSONString(const char *input, char *output,
  int output_size) {
  int output_index = 0;
  memset(output, 0, output_size);
  while (*input) {
    // Ensure that we have enough space for at least one full escaped value.
    if ((output_index - 2) >= output_size) break;
    // Block any non-ASCII characters
    if (*input >= 0x7f) {
      output[output_index] = '?';
      input++;
      output_index++;
      continue;
    }
    // Copy or escape acceptable characters.
    switch (*input) {
    // Backspace character
    case 0x08:
      output[output_index] = '\\';
      output_index++;
      output[output_index] = 'b';
      break;
    // Form feed character
    case 0x0c:
      output[output_index] = '\\';
      output_index++;
      output[output_index] = 'f';
      break;
    case '\r':
      output[output_index] = '\\';
      output_index++;
      output[output_index] = 'r';
      break;
    case '\t':
      output[output_index] = '\\';
      output_index++;
      output[output_index] = 't';
      break;
    case '\\':
      output[output_index] = '\\';
      output_index++;
      output[output_index] = '\\';
      break;
    case '"':
      output[output_index] = '\\';
      output_index++;
      output[output_index] = '"';
      break;
    case '\n':
      output[output_index] = '\\';
      output_index++;
      output[output_index] = 'n';
      break;
    default:
      output[output_index] = *input;
      break;
    }
    input++;
    output_index++;
  }
}

// Writes metadata to the output JSON file. Returns 0 on error.
static int WriteOutputHeader(PluginState *state) {
  DeviceInformation *info = &(state->shared_state->device_info);
  FILE *output = state->output_file;
  char buffer[SANITIZE_JSON_BUFFER_SIZE];
  if (fprintf(output, "{\n") < 0) {
    return 0;
  }
  SanitizeJSONString(state->shared_state->global_config->scenario_name,
    buffer, sizeof(buffer));
  if (fprintf(output, "\"scenario_name\": \"%s\",\n", buffer) < 0) {
    return 0;
  }
  SanitizeJSONString(state->functions.get_name(), buffer, sizeof(buffer));
  if (fprintf(output, "\"plugin_name\": \"%s\",\n", buffer) < 0) {
    return 0;
  }
  if (state->config->label) {
    SanitizeJSONString(state->config->label, buffer, sizeof(buffer));
    if (fprintf(output, "\"label\": \"%s\",\n", buffer) < 0) {
      return 0;
    }
  }
  if (fprintf(output, "\"release_time\": %f,\n",
    state->config->release_time) < 0) {
    return 0;
  }
  if (fprintf(output, "\"compute_unit_count\": %d,\n",
    info->compute_unit_count) < 0) {
    return 0;
  }
  if (fprintf(output, "\"threads_per_compute_unit\": %d,\n",
    info->threads_per_compute_unit) < 0) {
    return 0;
  }
  if (fprintf(output, "\"memory_clock_rate\": %d,\n",
    info->memory_clock_rate) < 0) {
    return 0;
  }
  if (fprintf(output, "\"warp_size\": %d,\n", info->warp_size) < 0) {
    return 0;
  }
  if (fprintf(output, "\"starting_clock\": %llu,\n",
    (unsigned long long) info->starting_clock) < 0) {
    return 0;
  }
  if (fprintf(output, "\"PID\": %d,\n", getpid()) < 0) {
    return 0;
  }
  // Only include the POSIX thread ID if threads are used.
  if (!state->shared_state->global_config->use_processes) {
    if (fprintf(output, "\"TID\": %d,\n", (int) GetThreadID()) < 0) {
      return 0;
    }
  }
  if (fprintf(output, "\"times\": [{}") < 0) {
    return 0;
  }
  fflush(output);
  return 1;
}

// Writes the start and end CPU times for this iteration to the output file.
static int WriteCPUTimesToOutput(PluginState *state, CPUTimes *t) {
  FILE *output = state->output_file;
  if (fprintf(output, ",\n{\"copy_in_times\": [%.9f,%.9f], ", t->copy_in_start,
    t->copy_in_end) < 0) {
    return 0;
  }
  if (fprintf(output, "\"execute_times\": [%.9f,%.9f], ", t->execute_start,
    t->execute_end) < 0) {
    return 0;
  }
  if (fprintf(output, "\"copy_out_times\": [%.9f,%.9f], ", t->copy_out_start,
    t->copy_out_end) < 0) {
    return 0;
  }
  // The total CPU time.
  if (fprintf(output, "\"cpu_times\": [%.9f,%.9f]}", t->copy_in_start,
    t->copy_out_end) < 0) {
    return 0;
  }
  return 1;
}

// Formats the given timing information as a JSON object and appends it to the
// output file. Returns 0 on error and 1 on success. Times will be written in a
// floatig-point number of *seconds*, even though they are recorded in ns. This
// code should not be included in plugin timing measurements.
static int WriteTimesToOutput(PluginState *state, TimingInformation *times) {
  SharedState *shared_state = state->shared_state;
  FILE *output = state->output_file;
  int i, j, block_time_count;
  char sanitized_name[SANITIZE_JSON_BUFFER_SIZE];
  uint64_t memory_clock_rate;
  uint64_t starting_clock;
  double t;
  KernelTimes *kernel_times = NULL;
  memory_clock_rate = state->shared_state->device_info.memory_clock_rate;
  starting_clock = state->shared_state->device_info.starting_clock;
  // Iterate over each kernel invocation
  for (i = 0; i < times->kernel_count; i++) {
    kernel_times = times->kernel_times + i;
    if (fprintf(output, ",\n{") < 0) {
      return 0;
    }
    // The kernel name may be NULL, but print it if it's provided.
    if (kernel_times->kernel_name) {
      sanitized_name[sizeof(sanitized_name) - 1] = 0;
      SanitizeJSONString(kernel_times->kernel_name, sanitized_name,
        sizeof(sanitized_name));
      if (fprintf(output, "\"kernel_name\": \"%s\", ", sanitized_name) < 0) {
        return 0;
      }
    }
    // Next, print this kernel's thread and block count.
    if (fprintf(output, "\"block_count\": %d, \"thread_count\": %d, ",
      kernel_times->block_count, kernel_times->thread_count) < 0) {
      return 0;
    }
    // Print the shared memory used by the kernel.
    if (fprintf(output, "\"shared_memory\": %d, ",
      kernel_times->shared_memory) < 0) {
      return 0;
    }
    // Print the kernel launch times for this kernel.
    if (fprintf(output, "\"kernel_launch_times\": [") < 0) {
      return 0;
    }
    // The time before the (CPU-side) kernel launch.
    t = kernel_times->kernel_launch_times[0] - shared_state->starting_seconds;
    if (fprintf(output, "%.9f, ", t) < 0) {
      return 0;
    }
    // The time after the kernel launch returned.
    t = kernel_times->kernel_launch_times[1] - shared_state->starting_seconds;
    if (fprintf(output, "%.9f, ", t) < 0) {
      return 0;
    }
    // The CPU time after the stream synchronize completed.
    if (kernel_times->kernel_launch_times[2] != 0) {
      t = kernel_times->kernel_launch_times[2] -
        shared_state->starting_seconds;
    } else {
      // We set this to 0 if the sync completion time wasn't recorded for this
      // kernel.
      t = 0;
    }
    if (fprintf(output, "%.9f], ", t) < 0) {
      return 0;
    }
    // Finally, include block times for the output. Try to convert them to
    // seconds using the reported clock rate. (TODO: Is the clock rate reported
    // by hipGetDeviceProperties actually going to lead to a steady time, or
    // can the clock rate change during kernel execution?)
    if (fprintf(output, "\"block_times\": [") < 0) {
      return 0;
    }
    // Remember we have both a start and end time for every block.
    block_time_count = kernel_times->block_count * 2;
    // Don't fill in the block times if the plugin didn't provide them.
    if (kernel_times->block_times == NULL) block_time_count = 0;
    for (j = 0; j < block_time_count; j++) {
      // The memory clock rate is in kHz.
      t = (double) (kernel_times->block_times[j] - starting_clock);
      t /= ((double) memory_clock_rate) * 1000.0;
      if (fprintf(output, "%.9f", t) < 0) {
        return 0;
      }
      if (j < (block_time_count - 1)) {
        if (fprintf(output, ",") < 0) {
          return 0;
        }
      }
    }

    // We're done printing information about this kernel, print the CPU core as
    // a sanity check.
    if (fprintf(output, "], \"cpu_core\": %d}", sched_getcpu()) < 0) {
      return 0;
    }
  }
  fflush(output);
  return 1;
}

// Returns the number of seconds elapsed since the global shared state was
// initialized.
static double GlobalSecondsElapsed(PluginState *state) {
  return CurrentSeconds() - state->shared_state->starting_seconds;
}

// Loads the plugin's shared library and finds the exported functions. Returns
// 0 on error. All other data in the PluginState struct must be initialized
// before calling this.
static int LoadPluginLibrary(PluginState *plugin) {
  RegisterPluginFunction register_plugin = NULL;
  plugin->library_handle = dlopen(plugin->config->filename, RTLD_NOW);
  if (!plugin->library_handle) {
    printf("Failed opening shared library %s: %s\n", plugin->config->filename,
      dlerror());
    return 0;
  }
  register_plugin = (RegisterPluginFunction) dlsym(plugin->library_handle,
    "RegisterPlugin");
  if (!register_plugin) {
    printf("The shared library %s doesn't export RegisterPlugin.\n",
      plugin->config->filename);
    dlclose(plugin->library_handle);
    plugin->library_handle = NULL;
    return 0;
  }
  if (!register_plugin(&(plugin->functions))) {
    printf("The shared library %s's RegisterFunctions function returned an "
      "error.\n", plugin->config->filename);
    dlclose(plugin->library_handle);
    plugin->library_handle = NULL;
    return 0;
  }
  return 1;
}

// Does a one or more non-recorded "Warm-up" iterations of a plugin. Returns 0
// on error. Does not call the cleanup function--the caller must do that if
// this returns 0.
static int DoWarmup(PluginState *state, void *user_data) {
  TimingInformation junk;
  static const int warmup_iterations = 4;
  const char *name = state->functions.get_name();
  int i;
  for (i = 0; i < warmup_iterations; i++) {
    if (!state->functions.copy_in(user_data)) {
      printf("Plugin %s copy in failed (warm-up).\n", name);
      return 0;
    }
    if (!state->functions.execute(user_data)) {
      printf("Plugin %s execute failed (warm-up).\n", name);
      return 0;
    }
    if (!state->functions.copy_out(user_data, &junk)) {
      printf("Plugin %s copy out failed (warm-up).\n", name);
      return 0;
    }
  }
  return 1;
}

// Runs a single plugin instance. This is usually called from a separate thread
// or process. Its argument must be a pointer to a PluginState struct. It may
// print a message and return NULL on error. On success, it will return an
// arbitrary non-NULL value. RegisterPlugin has already been called for the
// plugin, so the PluginFunctions struct has already been populated.
static void* RunPlugin(void *data) {
  InitializationParameters initialization_parameters;
  CPUTimes cpu_times;
  uint64_t i;
  int64_t max_iterations;
  double start_time, time_limit;
  PluginState *state = (PluginState *) data;
  ProcessBarrier *barrier = &(state->shared_state->barrier);
  int barrier_local_sense = 0;
  TimingInformation timing_info;
  void *user_data = NULL;
  const char *name;
  if (!LoadPluginLibrary(state)) {
    printf("Failed loading a plugin's shared library file.\n");
    return NULL;
  }
  name = state->functions.get_name();
  if (!InitializeInitializationParameters(state, &initialization_parameters)) {
    printf("Failed copying initialization parameters from config.\n");
    return NULL;
  }
  max_iterations = GetMaxIterations(state);
  time_limit = GetTimeLimit(state);
  if (!SetCPUAffinity(state)) {
    printf("Failed pinning instance of %s to a CPU core.\n", name);
    return NULL;
  }
  start_time = CurrentSeconds();
  user_data = state->functions.initialize(&initialization_parameters);
  if (!user_data) {
    printf("Failed initializing instance of %s.\n", name);
    return NULL;
  }
  if (!WriteOutputHeader(state)) {
    printf("Failed writing the output file header for %s.\n", name);
    return NULL;
  }
  // Do the warmup iteration(s) if required.
  if (state->shared_state->global_config->do_warmup) {
    if (!DoWarmup(state, user_data)) {
      state->functions.cleanup(user_data);
      return NULL;
    }
  }
  printf("Plugin %s initialized in %f seconds.\n", name, CurrentSeconds() -
    start_time);
  fflush(stdout);
  if (!BarrierWait(barrier, &barrier_local_sense)) {
    printf("Failed waiting for post-initialization synchronization.\n");
    state->functions.cleanup(user_data);
    return NULL;
  }
  // This function does nothing if the release time is 0 or lower.
  if (!SleepSeconds(state->config->release_time)) {
    state->functions.cleanup(user_data);
    return NULL;
  }
  i = 0;
  start_time = CurrentSeconds();
  while (1) {
    if (max_iterations > 0) {
      i++;
      if (i > max_iterations) break;
    }
    if (time_limit > 0) {
      if ((CurrentSeconds() - start_time) >= time_limit) break;
    }
    // If sync_every_iteration is true, we'll wait here for previous iterations
    // of all plugins to complete.
    if (state->shared_state->global_config->sync_every_iteration) {
      if (!BarrierWait(barrier, &barrier_local_sense)) {
        printf("Failed waiting to sync before an iteration.\n");
        state->functions.cleanup(user_data);
        return NULL;
      }
    }
    // A single plugin iteration consists of copy_in, execute, and copy_out.
    cpu_times.copy_in_start = GlobalSecondsElapsed(state);
    if (!state->functions.copy_in(user_data)) {
      printf("Plugin %s copy in failed.\n", name);
      state->functions.cleanup(user_data);
      return NULL;
    }
    cpu_times.copy_in_end = GlobalSecondsElapsed(state);
    cpu_times.execute_start = GlobalSecondsElapsed(state);
    if (!state->functions.execute(user_data)) {
      printf("Plugin %s execute failed.\n", name);
      state->functions.cleanup(user_data);
      return NULL;
    }
    cpu_times.execute_end = GlobalSecondsElapsed(state);
    cpu_times.copy_out_start = GlobalSecondsElapsed(state);
    if (!state->functions.copy_out(user_data, &timing_info)) {
      printf("Plugin %s copy out failed.\n", name);
      state->functions.cleanup(user_data);
      return NULL;
    }
    cpu_times.copy_out_end = GlobalSecondsElapsed(state);
    // Finally, write the timing data we obtained for this iteration to the
    // output file.
    if (!WriteCPUTimesToOutput(state, &cpu_times)) {
      printf("Failed writing CPU times for plugin %s to output file.\n", name);
      state->functions.cleanup(user_data);
      return NULL;
    }
    if (!WriteTimesToOutput(state, &timing_info)) {
      printf("Failed writing times for plugin %s to output file.\n", name);
      state->functions.cleanup(user_data);
      return NULL;
    }
  }
  // Wait before cleaning up any successful plugins in case cleaning up blocks
  // the GPU (in other cases, cleaning up occurred due to an error).
  if (!BarrierWait(barrier, &barrier_local_sense)) {
    printf("Failed waiting to sync before cleanup.\n");
    state->functions.cleanup(user_data);
    return NULL;
  }
  state->functions.cleanup(user_data);
  if (fprintf(state->output_file, "\n]}") < 0) {
    printf("Failed writing footer to output file.\n");
    return NULL;
  }
  return (void *) 1;
}

// Runs plugin instances in separate processes. Returns 1 on success and 0 if
// an error occurs. (Child processes may exit with a failure rather than
// returning.)
static int RunAsProcesses(SharedState *shared_state) {
  PluginState *plugins = shared_state->plugins;
  int i, child_status, all_ok, plugin_count;
  pid_t *pids = NULL;
  pid_t child_pid = 0;
  all_ok = 1;
  plugin_count = shared_state->global_config->plugin_count;
  pids = (pid_t *) malloc(plugin_count * sizeof(pid_t));
  if (!pids) {
    printf("Failed allocating space to hold PIDs.\n");
    return 0;
  }
  memset(pids, 0, plugin_count * sizeof(pid_t));
  for (i = 0; i < plugin_count; i++) {
    child_pid = fork();
    // The parent process can keep generating child processes.
    if (child_pid != 0) {
      pids[i] = child_pid;
      continue;
    }
    // The child process will run the plugin and exit with a success if
    // everything went OK.
    if (!RunPlugin(plugins + i)) {
      exit(EXIT_FAILURE);
    }
    exit(EXIT_SUCCESS);
  }
  // As the parent, ensure that each child exited and exited with EXIT_SUCCESS.
  for (i = 0; i < plugin_count; i++) {
    waitpid(pids[i], &child_status, 0);
    if (!WIFEXITED(child_status)) {
      printf("A child process ended without exiting properly.\n");
      all_ok = 0;
    } else if (WEXITSTATUS(child_status) != EXIT_SUCCESS) {
      printf("A child process exited with an error.\n");
      all_ok = 0;
    }
  }
  free(pids);
  return all_ok;
}

// Runs plugin instances in threads. Returns 1 on success and 0 if an error
// occurs.
static int RunAsThreads(SharedState *shared_state) {
  PluginState *plugins = shared_state->plugins;
  pthread_t *threads = NULL;
  int i, result, to_return, plugin_count;
  void *thread_result;
  plugin_count = shared_state->global_config->plugin_count;
  threads = (pthread_t *) malloc(plugin_count * sizeof(pthread_t));
  if (!threads) {
    printf("Failed allocating space to hold thread IDs.\n");
    return 0;
  }
  memset(threads, 0, plugin_count * sizeof(pthread_t));
  to_return = 1;
  for (i = 0; i < plugin_count; i++) {
    result = pthread_create(threads + i, NULL, RunPlugin, plugins + i);
    if (result != 0) {
      printf("Failed starting a thread: %d\n", result);
      to_return = 0;
      break;
    }
  }
  // Wait on threads in reverse order, in case not all of them were created.
  i--;
  for (; i >= 0; i--) {
    result = pthread_join(threads[i], &thread_result);
    if (result != 0) {
      printf("Failed joining thread for plugin %d: %d\n", i, result);
      to_return = 0;
      continue;
    }
    if (!thread_result) {
      printf("A child thread exited with an error.\n");
      to_return = 0;
    }
  }
  free(threads);
  return to_return;
}

int main(int argc, char **argv) {
  int result;
  char *config_content = NULL;
  SharedState *shared_state = NULL;
  if (argc != 2) {
    printf("Usage: %s <path to JSON config file>\n", argv[0]);
    return 1;
  }
  // Read the config file, then set up the top-level SharedState struct.
  config_content = (char *) GetConfigFileContent(argv[1]);
  if (!config_content) return 1;
  shared_state = (SharedState *) malloc(sizeof(*shared_state));
  if (!shared_state) {
    printf("Failed allocating shared state buffer.\n");
    return 1;
  }
  memset(shared_state, 0, sizeof(*shared_state));
  shared_state->global_config = ParseConfiguration(config_content);
  if (!shared_state->global_config) {
    Cleanup(shared_state);
    return 1;
  }
  // The original config text isn't needed anymore after parsing it.
  free(config_content);
  config_content = NULL;
  if (!CreatePluginStates(shared_state)) {
    Cleanup(shared_state);
    return 1;
  }
  printf("Plugin configuration loaded.\n");
  if (!BarrierCreate(&(shared_state->barrier),
    shared_state->global_config->plugin_count)) {
    printf("Failed initializing synchronization barrier.\n");
    Cleanup(shared_state);
    return 1;
  }
  shared_state->barrier_created = 1;
  if (!GetDeviceInfo(shared_state->global_config->gpu_device_id,
    &(shared_state->device_info))) {
    printf("Failed reading device information.\n");
    Cleanup(shared_state);
    return 1;
  }
  printf("Running on device %d: %s\n",
    shared_state->global_config->gpu_device_id,
    shared_state->device_info.device_name);
  shared_state->starting_seconds = CurrentSeconds();
  if (shared_state->global_config->use_processes) {
    result = RunAsProcesses(shared_state);
  } else {
    result = RunAsThreads(shared_state);
  }
  if (!result) {
    printf("An error occurred executing one or more plugins.\n");
  } else {
    printf("All plugins completed successfully.\n");
  }
  Cleanup(shared_state);
  return 0;
}
