// This plugin allocates two vectors of floating-point numbers.
// The additional_info must be a JSON object with the following keys:
//
//  - "vector_length": The number of floating-point values in the two vectors
//    to add.
//
// - "skip_copy": A boolean. Optional, defaults to false. If it's set to true,
//   then the "copy_in" and "copy_out" phases will not copy the vector data to
//   or from the GPU. Instead, the input vectors will only be copied once,
//   during the initialize function, and output data will never be copied. May
//   be useful if you want a simple workload without so many memory transfers.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include "third_party/cJSON.h"
#include "plugin_hip_utilities.h"
#include "plugin_interface.h"
#include "plugin_utilities.h"

// Holds the state of an instance of this plugin.
typedef struct {
  hipStream_t stream;
  // Tracking whether the stream is created simplifies cleanup code.
  int stream_created;
  // The block_count is computed based on the vector size.
  int block_count;
  // The thread count is specified by the config.
  int thread_count;
  // Set this to nonzero in order to not copy input or output vectors during
  // copy_in or copy_out. Instead, input vectors will only be copied once,
  // during initialization.
  int skip_copy;
  // The number of floating-point values in each vector.
  uint64_t vector_length;
  // The device's input vectors a and b, and result vector c.
  float *d_a, *d_b, *d_c;
  // The three vectors, except on the host.
  float *h_a, *h_b, *h_c;
  // The recordings of the start and end GPU clock cycle for each block.
  uint64_t *device_block_times;
  // Holds times that are shared with the plugin host.
  KernelTimes kernel_times;
} PluginState;

// Implements the Cleanup() function required by the plugin interface.
static void Cleanup(void *data) {
  PluginState *state = (PluginState *) data;
  if (!state) return;
  hipFree(state->d_a);
  hipFree(state->d_b);
  hipFree(state->d_c);
  hipHostFree(state->h_a);
  hipHostFree(state->h_b);
  hipHostFree(state->h_c);
  hipHostFree(state->kernel_times.block_times);
  if (state->stream_created) {
    CheckHIPError(hipStreamDestroy(state->stream));
  }
  memset(state, 0, sizeof(*state));
  free(state);
}

// Allocates the host and device vectors, and initializes the input vectors.
// Returns 0 on error.
static int AllocateMemory(PluginState *state) {
  uint64_t i;
  size_t size;

  // Allocate host and device memory for block times.
  size = state->block_count * 2 * sizeof(uint64_t);
  if (!CheckHIPError(hipHostMalloc(&(state->kernel_times.block_times),
    size))) {
    return 0;
  }
  if (!CheckHIPError(hipMalloc(&(state->device_block_times), size))) {
    return 0;
  }

  // Allocate the vectors.
  size = state->vector_length * sizeof(float);
  if (!CheckHIPError(hipMalloc(&state->d_a, size))) return 0;
  if (!CheckHIPError(hipMalloc(&state->d_b, size))) return 0;
  if (!CheckHIPError(hipMalloc(&state->d_c, size))) return 0;
  if (!CheckHIPError(hipHostMalloc(&state->h_a, size))) return 0;
  if (!CheckHIPError(hipHostMalloc(&state->h_b, size))) return 0;
  if (!CheckHIPError(hipHostMalloc(&state->h_c, size))) return 0;

  // Randomly initialize the host's input vectors.
  for (i = 0; i < state->vector_length; i++) {
    // Obviously this is a strange way to make random floats, but that
    // shouldn't matter for this simple microbenchmark.
    state->h_a[i] = (float) rand();
    state->h_b[i] = (float) rand();
  }

  // Initialize input vectors on the device.
  if (!CheckHIPError(hipMemcpyAsync(state->d_a, state->h_a, size,
    hipMemcpyHostToDevice, state->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipMemcpyAsync(state->d_b, state->h_b, size,
    hipMemcpyHostToDevice, state->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  return 1;
}

// Parses the additional_info argument. Returns 0 on error.
static int ParseAdditionalInfo(const char *arg, PluginState *state) {
  cJSON *root = NULL;
  cJSON *entry = NULL;
  root = cJSON_Parse(arg);
  if (!root) {
    printf("Invalid additional_info for vector_add.\n");
    return 0;
  }

  // Make sure that vector_length is present and positive.
  entry = cJSON_GetObjectItem(root, "vector_length");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Invalid vector_length setting.\n");
    cJSON_Delete(root);
    return 0;
  }
  if (entry->valuedouble < 1.0) {
    printf("The vector_length setting must be at least 1.\n");
    cJSON_Delete(root);
    return 0;
  }
  if (entry->valuedouble > 1e32) {
    printf("Warning: huge vector length: %f\n", entry->valuedouble);
  }
  state->vector_length = (uint64_t) entry->valuedouble;

  // If skip_copy is present, make sure it's a boolean. state->skip_copy is
  // initialized to 0 already.
  entry = cJSON_GetObjectItem(root, "skip_copy");
  if (entry) {
    if ((entry->type != cJSON_True) && (entry->type != cJSON_False)) {
      printf("The skip_copy setting must be a boolean.\n");
      cJSON_Delete(root);
      return 0;
    }
    state->skip_copy = entry->type == cJSON_True;
  }

  cJSON_Delete(root);
  root = NULL;
  entry = NULL;
  return 1;
}

static void* Initialize(InitializationParameters *params) {
  PluginState *state = NULL;
  int block_count, thread_count;
  uint64_t vector_length;
  if (!CheckHIPError(hipSetDevice(params->device_id))) return NULL;
  if (!CheckHIPError(hipSetDeviceFlags(PLUGIN_DEVICE_FLAGS))) return NULL;
  state = (PluginState *) calloc(1, sizeof(*state));
  if (!state) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }

  // We need to know the vector length before we can allocate memory or
  // determine block size.
  if (!ParseAdditionalInfo(params->additional_info, state)) {
    Cleanup(state);
    return NULL;
  }
  vector_length = state->vector_length;

  // Compute the block count. Make sure to add an additional block if the
  // vector length isn't evenly divisible by the thread count.
  if (!GetSingleBlockDimension(params, &thread_count)) {
    Cleanup(state);
    return NULL;
  }
  block_count = (int) (vector_length / ((uint64_t) thread_count));
  if ((thread_count > vector_length) &&
    ((vector_length % thread_count) != 0)) {
    block_count++;
  }
  state->thread_count = thread_count;
  state->block_count = block_count;

  // Create the stream and fill in boilerplate for reporting to the framework.
  if (!CheckHIPError(CreateHIPStreamWithMask(&(state->stream),
    params->compute_unit_mask, COMPUTE_UNIT_MASK_ENTRIES))) {
    Cleanup(state);
    return NULL;
  }
  state->stream_created = 1;
  state->kernel_times.kernel_name = "vector_add";
  state->kernel_times.thread_count = state->thread_count;
  state->kernel_times.block_count = state->block_count;
  state->kernel_times.shared_memory = 0;

  // Allocate the vectors and initialize the input vectors.
  if (!AllocateMemory(state)) {
    Cleanup(state);
    return NULL;
  }
  return state;
}

// Reset block times and copy input vectors.
static int CopyIn(void *data) {
  PluginState *state = (PluginState *) data;

  // Reset block times
  size_t size = state->block_count * 2 * sizeof(uint64_t);
  if (!CheckHIPError(hipMemsetAsync(state->device_block_times, 0xff,
    size, state->stream))) {
    return 0;
  }

  // If we're skipping the copies we can return now.
  if (state->skip_copy) {
    if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
    return 1;
  }

  // Copy input vectors
  size = state->vector_length * sizeof(float);
  if (!CheckHIPError(hipMemcpyAsync(state->d_a, state->h_a, size,
    hipMemcpyHostToDevice, state->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipMemcpyAsync(state->d_b, state->h_b, size,
    hipMemcpyHostToDevice, state->stream))) {
    return 0;
  }

  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  return 1;
}

// The kernel for actually adding the buffers. Sets c = a + b.
__global__ void VectorAddKernel(float *a, float *b, float *c, uint64_t length,
  uint64_t *block_times) {
  uint64_t i;
  uint64_t start_clock = clock64();
  if (start_clock < block_times[blockIdx.x * 2]) {
    block_times[blockIdx.x * 2] = start_clock;
  }

  // Add the actual vector element.
  i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < length) {
    c[i] = a[i] + b[i];
  }

  block_times[blockIdx.x * 2 + 1] = clock64();
}

static int Execute(void *data) {
  PluginState *state = (PluginState *) data;
  state->kernel_times.kernel_launch_times[0] = CurrentSeconds();

  hipLaunchKernelGGL(VectorAddKernel, state->block_count, state->thread_count,
    0, state->stream, state->d_a, state->d_b, state->d_c, state->vector_length,
    state->device_block_times);

  state->kernel_times.kernel_launch_times[1] = CurrentSeconds();
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  state->kernel_times.kernel_launch_times[2] = CurrentSeconds();
  return 1;
}

// Copy the block times to the host, along with the result vector.
static int CopyOut(void *data, TimingInformation *times) {
  PluginState *state = (PluginState *) data;
  size_t size;

  // First copy the block times to the host.
  size = state->block_count * 2 * sizeof(uint64_t);
  if (!CheckHIPError(hipMemcpyAsync(state->kernel_times.block_times,
    state->device_block_times, size, hipMemcpyDeviceToHost, state->stream))) {
    return 0;
  }

  // Provide the framework with a pointer to our kernel_times struct.
  times->kernel_count = 1;
  times->kernel_times = &(state->kernel_times);

  // If we're skipping copying the vectors, we can return now.
  if (state->skip_copy) {
    times->resulting_data_size = 0;
    times->resulting_data = NULL;
    if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
    return 1;
  }

  // Next, copy the resulting vector.
  size = state->vector_length * sizeof(float);
  if (!CheckHIPError(hipMemcpyAsync(state->h_c, state->d_c, size,
    hipMemcpyDeviceToHost, state->stream))) {
    return 0;
  }
  times->resulting_data_size = size;
  times->resulting_data = state->h_c;
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  return 1;
}

static const char* GetName(void) {
  return "Vector Add";
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

