// This file defines the tool used for launching HIP plugins, contained in
// shared library files, as either threads or processes. Supported shared
// libraries must implement the RegisterFunctions(...) function as defined in
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "barrier_wait.h"
#include "parse_config.h"
#include "plugin_interface.h"

// Config files are read in chunks containing this many bytes (useful for
// reading from stdin).
#define FILE_CHUNK_SIZE (4096)

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
} SharedState;

// Holds configuration data specific to a given plugin.
typedef struct PluginState_ {
  // A reference to the PluginConfiguration for this plugin. Do not free this,
  // it will be freed when cleaning up the global_config object in SharedState.
  PluginConfiguration *config;
  PluginFunctions functions;
  InitializationParameters parameters;
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
typedef int (*RegisterFunctionsFunction)(PluginFunctions *functions);

// Returns a floating-point number of seconds from the system clock.
static double CurrentSeconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    printf("Error getting time.\n");
    exit(1);
  }
  return ((double) ts.tv_sec) + (((double) ts.tv_nsec) / 1e9);
}

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

// Allocates and initializes the plugins list in shared_state. The global
// configuration must have already been loaded for this to succeed. Returns 0
// on error.
static int CreatePluginStates(SharedState *shared_state) {
  RegisterFunctionsFunction register_functions = NULL;
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
    // Finally, open the shared library and get the function pointers.
    new_list[i].library_handle = dlopen(plugin_config->filename, RTLD_NOW);
    if (!new_list[i].library_handle) {
      printf("Failed opening shared library %s: %s\n", plugin_config->filename,
        dlerror());
      fflush(stdout);
      goto ErrorCleanup;
    }
    register_functions = (RegisterFunctionsFunction) dlsym(
      new_list[i].library_handle, "RegisterFunctions");
    if (!register_functions) {
      printf("The shared library %s doesn't export RegisterFunctions.\n",
        plugin_config->filename);
      goto ErrorCleanup;
    }
    if (!register_functions(&(new_list[i].functions))) {
      printf("The shared library %s's RegisterFunctions returned an error.\n",
        plugin_config->filename);
      goto ErrorCleanup;
    }
  }
  shared_state->plugins = new_list;
  return 1;
ErrorCleanup:
  for (i = 0; i < config->plugin_count; i++) {
    if (new_list[i].output_file) fclose(new_list[i].output_file);
    if (new_list[i].library_handle) dlclose(new_list[i].library_handle);
  }
  free(new_list);
  return 0;
}

int main(int argc, char **argv) {
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
  shared_state->starting_seconds = CurrentSeconds();
  // TODO: Actually run the plugins.
  return 0;
}
