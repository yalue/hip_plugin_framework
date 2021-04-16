// This plugin is intended to test locking overheads by manually locking the
// GPU with high priority. Do *not* use this with the job_deadline config
// option, as it will set its own deadline. It is an error to use this plugin
// if the gpu_locking_module isn't available. Do *not* use this when
// use_processes is false; behavior would not be particularly useful and it's
// likely that GPU-using tasks would never finish.
//
// The additional_info object in the config is currently optional, omitting it
// falls back to the defaults. It may contain the following keys:
//
// - "preempt_count": Defaults to 1. If provided, must contain a non-negative
//   integer specifying the number of times to acquire and release the lock,
//   under the assumption that each acquire/release will preempt other GPU
//   work. Note that setting this to 0 *is valid*, and the execute phase of the
//   plugin will basically do nothing.

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "third_party/cJSON.h"
#include "gpu_locking_module.h"
#include "plugin_hip_utilities.h"
#include "plugin_interface.h"
#include "plugin_utilities.h"

typedef struct {
  // The number of times to attempt to acquire and release the lock per
  // Execute(...) invocation.
  int preempt_count;
  // The descriptor for the gpu_locking_module chardev.
  int fd;
  // The ID of the lock we'll acquire and release. Set using the GPU_LOCK_ID
  // environment variable (also used by my modified HIP runtime).
  int lock_id;
} PluginState;

// Implements the Cleanup() function required by the plugin interface.
static void Cleanup(void *data) {
  PluginState *state = (PluginState *) data;
  if (state->fd < 0) {
    close(state->fd);
    state->fd = -1;
  }
  free(state);
}

// We'll launch this kernel during initialization simply to ensure that we've
// used HIP and it's loaded. Sets *out to the GPU's current clock64.
__global__ void DummyKernel(uint64_t *out) {
  *out = clock64();
}

// Runs DummyKernel to ensure that HIP has been loaded. Returns 0 on error.
// Takes an "output" pointer to fill, to ensure it won't get optimized out.
static int RunDummyKernel(uint64_t *output) {
  uint64_t *gpu_clock = NULL;
  if (!CheckHIPError(hipMalloc(&gpu_clock, sizeof(uint64_t)))) return 0;
  hipLaunchKernelGGL(DummyKernel, 1, 1, 0, 0, gpu_clock);
  if (!CheckHIPError(hipDeviceSynchronize())) {
    hipFree(gpu_clock);
    return 0;
  }
  if (!CheckHIPError(hipMemcpy(output, gpu_clock, sizeof(uint64_t),
    hipMemcpyDeviceToHost))) {
    hipFree(gpu_clock);
    return 0;
  }
  hipFree(gpu_clock);
  return 1;
}

// Parses the additional info, if it is present. Returns 0 on error. Silently
// returns success if nothing was specified.
static int ParseAdditionalInfo(const char *arg, PluginState *state) {
  cJSON *root = NULL;
  cJSON *entry = NULL;
  // Set the default values.
  state->preempt_count = 1;
  // Allow additional_info to be omitted if we want to take all the defaults
  if (!arg || (strlen(arg) == 0)) return 1;

  root = cJSON_Parse(arg);
  if (!root) {
    printf("Invalid additional_info for dummy_lock_gpu.\n");
    return 0;
  }
  if (root->type != cJSON_Object) {
    printf("The additional_info for dummy_lock_gpu must be a JSON object.\n");
    cJSON_Delete(root);
    return 0;
  }
  entry = cJSON_GetObjectItem(root, "preempt_count");
  if (entry) {
    if (entry->type != cJSON_Number) {
      printf("Invalid preempt_count setting for dummy_lock_gpu.\n");
      cJSON_Delete(root);
      return 0;
    }
    if (entry->valueint < 0) {
      printf("The preempt_count for dummy_lock_gpu can't be negative.\n");
      cJSON_Delete(root);
      return 0;
    }
    state->preempt_count = entry->valueint;
  }
  cJSON_Delete(root);
  return 1;
}

// Checks for the GPU_LOCK_ID environment variable and sets *id to its contents
// if it's present and valid-seeming. Returns 1 on success. Returns 0 on error,
// including if the environment variable contains an invalid value. Sets *id to
// 0 by default.
static int GetLockID(int *id) {
  char *s = getenv("GPU_LOCK_ID");
  char *end = NULL;
  int v = 0;
  *id = 0;
  if (!s) return 1;
  v = (int) strtol(s, &end, 10);
  // We won't count it as an error if the environment variable is simply empty.
  if (s == end) return 1;

  // It's an error if the environment variable wasn't empty, but contained a
  // non-digit, or contained a negative number.
  if ((*end != 0) || (v < 0)) {
    printf("Invalid GPU_LOCK_ID environment variable.\n");
    return 0;
  }

  // If the environment variable contained a lock ID that is too large we'll
  // just let the kernel module return EINVAL.
  *id = (int) v;
  return 1;
}

static void* Initialize(InitializationParameters *params) {
  PluginState *state = NULL;
  SetDeadlineArgs ioctl_args;
  uint64_t tmp;
  int result;
  // We will initialize and use HIP *before* opening the chardev, by launching
  // a kernel to ensure HIP has been loaded.
  if (!CheckHIPError(hipSetDevice(params->device_id))) return NULL;
  if (!CheckHIPError(hipSetDeviceFlags(PLUGIN_DEVICE_FLAGS))) return NULL;
  // Make it "possible" to use the output to prevent optimizing away the kernel
  if (!RunDummyKernel(&tmp)) return NULL;
  if (CurrentSeconds() == 0) printf("%ld\n", (long) tmp);

  state = (PluginState *) calloc(1, sizeof(*state));
  if (!state) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }
  state->fd = -1;
  if (!ParseAdditionalInfo(params->additional_info, state)) {
    Cleanup(state);
    return NULL;
  }
  state->fd = open("/dev/gpu_locking_module", O_RDWR);
  if (state->fd < 0) {
    printf("The dummy_lock_gpu plugin requires /dev/gpu_locking_module, but "
      "couldn't open it: %s\n", strerror(errno));
    Cleanup(state);
    return NULL;
  }
  // We'll set our deadline to a very high priority *now*, and never update it.
  // Since deadlines are relative, this should virtually ensure that all of our
  // subsequent lock requests are top-priority, as other tasks probably won't
  // be setting deadlines during Initialize(...)
  ioctl_args.deadline = 1;
  result = ioctl(state->fd, GPU_LOCK_SET_DEADLINE_IOC, &ioctl_args);
  if (result != 0) {
    printf("The dummy_lock_gpu plugin failed setting priority: %s\n",
      strerror(errno));
    Cleanup(state);
    return NULL;
  }
  if (!GetLockID(&(state->lock_id))) {
    printf("Failed getting the lock ID for the dummy_lock_gpu plugin.\n");
    Cleanup(state);
    return NULL;
  }
  return state;
}

static int CopyIn(void *data) {
  return 1;
}

static void ShortSleep(void) {
  struct timespec ts;
  memset(&ts, 0, sizeof(ts));
  // We'll sleep for approximately one microsecond.
  ts.tv_nsec = 1000;
  nanosleep(&ts, NULL);
}

static int Execute(void *data) {
  PluginState *state = (PluginState *) data;
  GPULockArgs ioctl_args;
  int i, result;
  ioctl_args.lock_id = state->lock_id;

  for (i = 0; i < state->preempt_count; i++) {
    // Do a tiny sleep between preempts to give whatever's being preempted time
    // to get back onto the GPU.
    ShortSleep();
    result = ioctl(state->fd, GPU_LOCK_ACQUIRE_IOC, &ioctl_args);
    if (result != 0) {
      printf("Failed acquiring GPU lock: %s\n", strerror(errno));
      return 0;
    }
    result = ioctl(state->fd, GPU_LOCK_RELEASE_IOC, &ioctl_args);
    if (result != 0) {
      printf("Failed releasing GPU lock: %s\n", strerror(errno));
      return 0;
    }
  }
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  // We didn't launch any kernels.
  times->kernel_count = 0;
  times->kernel_times = NULL;
  times->resulting_data_size = 0;
  times->resulting_data = NULL;
  return 1;
}

static const char* GetName(void) {
  return "Dummy Lock GPU";
}

int RegisterPlugin(PluginFunctions *functions) {
  functions->get_name = GetName;
  functions->cleanup = Cleanup;
  functions->initialize = Initialize;
  functions->copy_in = CopyIn;
  functions->execute = Execute;
  functions->copy_out = CopyOut;
  return 1;
}

