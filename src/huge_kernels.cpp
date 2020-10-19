// This plugin includes several "huge" kernels, created using inline assembly
// to involve a large number of instructions. The intent behind this plugin is
// to stress the L1 instruction cache.
//
// This kernel takes an "additional_info" parameter: a JSON object with two
// entries. One is "kernel_selection", which must be a string specifying which
// of the supported kernels to use. It, however, is currently optional and only
// "add" is supported. The second entry is "repetitions", which indicates the
// number of times the large number of instructions in each kernel is to be
// repeated. This is also optional and defaults to 1. If the additional_info
// parameter is omitted entirely, both of these settings take their default
// values.  Example:
//
// "additional_info": {
//   "kernel_selection": "add",
//   "repetitions": 1000,
// }

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include "plugin_hip_utilities.h"
#include "plugin_interface.h"
#include "plugin_utilities.h"
#include "third_party/cJSON.h"

// We'll abuse these macros to generate a lot of inline assembly.
#define REPEAT_10X(v) v; \
        v; \
        v; \
        v; \
        v; \
        v; \
        v; \
        v; \
        v; \
        v;

#define REPEAT_100X(v) REPEAT_10X(REPEAT_10X(v))

#define REPEAT_10000X(v) REPEAT_100X(REPEAT_100X(v))

// This specifies the kernel that will be run.
typedef enum {
  HUGE_ADD_KERNEL = 0,
} KernelSelection;

typedef struct {
  hipStream_t stream;
  int stream_created;
  int block_count;
  int thread_count;
  // kernel_times and device_block_times are just the bookkeeping data similar
  // to most plugins.
  KernelTimes kernel_times;
  uint64_t *device_block_times;
  // Our kernels produce one floating-point output per GPU thread.
  float *device_results;
  float *host_results;
  // This will be set at initialization--just an additional parameter to the
  // kernel.
  float arbitrary_value;
  // This will specify the number of times to repeat a loop in the kernel, to
  // make the kernel take longer, continuing the L1 I-cache pressure.
  uint64_t kernel_repetitions;
  // The kernel to run for this instance of the plugin.
  KernelSelection kernel_selection;
} PluginState;

static void Cleanup(void *data) {
  PluginState *s = (PluginState *) data;
  if (!s) return;
  hipFree(s->device_block_times);
  hipHostFree(s->kernel_times.block_times);
  hipFree(s->device_results);
  hipHostFree(s->host_results);
  memset(s, 0, sizeof(*s));
  free(s);
}

// Parses the additional_info argument. In this plugin, this function checks
// that a valid kernel has been specified and fills in the kernel_selection
// field appropriately. Returns 0 on error.
static int ParseAdditionalInfo(const char *arg, PluginState *s) {
  cJSON *root = NULL;
  cJSON *entry = NULL;
  s->kernel_selection = HUGE_ADD_KERNEL;
  s->kernel_repetitions = 1;
  if (!arg) {
    return 1;
  }
  root = cJSON_Parse(arg);
  if (!root) {
    printf("Invalid additional_info for huge_kernels\n");
    return 0;
  }
  entry = cJSON_GetObjectItem(root, "kernel_selection");
  if (entry) {
    if (entry->type != cJSON_String) {
      printf("The kernel_selection for huge_kernels must be a string.\n");
      cJSON_Delete(root);
      return 0;
    }
    if (strcmp(entry->valuestring, "add") != 0) {
      printf("The huge_kernels plugin only currently supports an \"add\" "
        "kernel, got \"%s\"\n", entry->valuestring);
      cJSON_Delete(root);
      return 0;
    }
  }
  entry = cJSON_GetObjectItem(root, "repetitions");
  if (entry) {
    if (entry->type != cJSON_Number) {
      printf("The repetitions setting for huge_kernels must be a number.\n");
      cJSON_Delete(root);
      return 0;
    }
    // We're using valuedouble in case the setting overflows an int.
    if (entry->valuedouble < 1.0) {
      printf("The repetitions setting for huge_kernels must be positive.\n");
      cJSON_Delete(root);
      return 0;
    }
    s->kernel_repetitions = (uint64_t) entry->valuedouble;
  }
  cJSON_Delete(root);
  return 1;
}

// Allocates host and device memory for the plugin. Expects the number of
// blocks and threads to be set already. Returns 0 on error.
static int AllocateMemory(PluginState *s) {
  size_t size = s->block_count * 2 * sizeof(uint64_t);
  if (!CheckHIPError(hipHostMalloc(&(s->kernel_times.block_times), size))) {
    return 0;
  }
  if (!CheckHIPError(hipMalloc(&s->device_block_times, size))) return 0;
  size = s->block_count * s->thread_count * sizeof(float);
  if (!CheckHIPError(hipHostMalloc(&s->host_results, size))) return 0;
  if (!CheckHIPError(hipMalloc(&s->device_results, size))) return 0;
  return 1;
}

static void* Initialize(InitializationParameters *params) {
  PluginState *s = NULL;
  if (!CheckHIPError(hipSetDevice(params->device_id))) return NULL;
  if (!CheckHIPError(hipSetDeviceFlags(PLUGIN_DEVICE_FLAGS))) return NULL;
  s = (PluginState *) malloc(sizeof(*s));
  if (!s) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }
  memset(s, 0, sizeof(*s));
  s->arbitrary_value = (float) CurrentSeconds();
  s->block_count = params->block_count;
  s->thread_count = params->thread_count;
  if (!ParseAdditionalInfo(params->additional_info, s)) {
    Cleanup(s);
    return NULL;
  }
  if (!AllocateMemory(s)) {
    Cleanup(s);
    return NULL;
  }
  if (!CheckHIPError(CreateHIPStreamWithMask(&s->stream,
    params->compute_unit_mask, COMPUTE_UNIT_MASK_ENTRIES))) {
    Cleanup(s);
    return NULL;
  }
  s->stream_created = 1;
  switch (s->kernel_selection) {
    case HUGE_ADD_KERNEL:
      s->kernel_times.kernel_name = "HugeAddKernel";
      break;
    default:
      printf("Invalid kernel selection.\n");
      Cleanup(s);
      return NULL;
  }
  s->kernel_times.thread_count = s->thread_count;
  s->kernel_times.block_count = s->block_count;
  s->kernel_times.shared_memory = 0;
  return s;
}

static int CopyIn(void *data) {
  PluginState *s = (PluginState *) data;
  // The device block times must hold the max 64-bit int value for the
  // algorithm for setting the earliest time to work.
  if (!CheckHIPError(hipMemsetAsync(s->device_block_times, 0xff,
    2 * s->block_count * sizeof(uint64_t), s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipMemsetAsync(s->device_results, 0,
    s->thread_count * s->block_count * sizeof(float), s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
  return 1;
}

__global__ void HugeAddKernel(float *results, float increment,
  uint64_t repetitions, uint64_t *block_times) {
  uint64_t start_time = clock64();
  if (start_time < block_times[blockIdx.x * 2]) {
    block_times[blockIdx.x * 2] = start_time;
  }
  float accumulator = 0;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t n = 0;

  for (n = 0; n < repetitions; n++) {
    REPEAT_10000X(asm volatile ("v_add_f32 %0, %1, %2" : "=v"(accumulator) :
      "v"(increment), "v"(accumulator)));
    REPEAT_10000X(asm volatile ("v_add_f32 %0, %1, %2" : "=v"(accumulator) :
      "v"(increment), "v"(accumulator)));
    REPEAT_10000X(asm volatile ("v_add_f32 %0, %1, %2" : "=v"(accumulator) :
      "v"(increment), "v"(accumulator)));
  }
  __syncthreads();

  results[index] = accumulator;
  block_times[blockIdx.x * 2 + 1] = clock64();
}

static int Execute(void *data) {
  PluginState *s = (PluginState *) data;
  s->kernel_times.kernel_launch_times[0] = CurrentSeconds();
  switch (s->kernel_selection) {
    case (HUGE_ADD_KERNEL):
      hipLaunchKernelGGL(HugeAddKernel, s->block_count, s->thread_count, 0,
        s->stream, s->device_results, s->arbitrary_value,
        s->kernel_repetitions, s->device_block_times);
      break;
    default:
      // This should never be possible, because it should be caught during the
      // Initialize function...
      printf("Internal error: invalid kernel selection.\n");
      return 0;
  }
  s->kernel_times.kernel_launch_times[1] = CurrentSeconds();
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
  s->kernel_times.kernel_launch_times[2] = CurrentSeconds();
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  PluginState *s = (PluginState *) data;
  size_t size = s->block_count * 2 * sizeof(uint64_t);
  if (!CheckHIPError(hipMemcpyAsync(s->kernel_times.block_times,
    s->device_block_times, size, hipMemcpyDeviceToHost, s->stream))) {
    return 0;
  }
  size = s->block_count * s->thread_count * sizeof(float);
  if (!CheckHIPError(hipMemcpyAsync(s->host_results, s->device_results, size,
    hipMemcpyDeviceToHost, s->stream))) {
    return 0;
  }
  times->kernel_count = 1;
  times->kernel_times = &(s->kernel_times);
  times->resulting_data = s->host_results;
  times->resulting_data_size = s->thread_count * s->block_count *
    sizeof(float);
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
  return 1;
}

static const char* GetName(void) {
  return "Huge Kernels";
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

