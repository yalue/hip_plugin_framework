// This plugin probes a buffer of GPU memory in random order, to stress GPU
// memory and cache.  The "additional_info" field contains a JSON object
// specifying the "buffer_size" in bytes, and "iterations"--the number of times
// each thread will probe memory.
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
  // This plugin can use an arbitrary number of threads and blocks.
  int block_count;
  int thread_count;
  // The number of loop iterations each thread must spin for.
  uint64_t iterations;
  // The size of the random-walk buffer, as a number of 64-bit integers.
  uint64_t buffer_length;
  uint64_t *device_random_buffer;
  uint64_t *device_block_times;
  // Holds times that are shared with the plugin host.
  KernelTimes kernel_times;
} PluginState;

// Returns a single random 64-bit value.
static uint64_t Random64(void) {
  int i;
  uint64_t to_return = 0;
  // Get a random number in 16-bit chunks
  for (i = 0; i < 4; i++) {
    to_return = to_return << 16;
    to_return |= rand() & 0xffff;
  }
  return to_return;
}

// Returns a random 64-bit integer in the range [base, limit)
static uint64_t RandomRange(uint64_t base, uint64_t limit) {
  if (limit <= base) return base;
  return (Random64() % (limit - base)) + base;
}

// Shuffles an array of 32-bit values.
static void ShuffleArray(uint64_t *buffer, uint64_t element_count) {
  uint64_t tmp;
  uint64_t i, dst;
  for (i = 0; i < element_count; i++) {
    dst = RandomRange(i, element_count);
    tmp = buffer[i];
    buffer[i] = buffer[dst];
    buffer[dst] = tmp;
  }
}

// Implements the Cleanup() function required by the plugin interface.
static void Cleanup(void *data) {
  PluginState *state = (PluginState *) data;
  if (!state) return;
  hipFree(state->device_block_times);
  hipHostFree(state->kernel_times.block_times);
  hipHostFree(state->device_random_buffer);
  if (state->stream_created) {
    CheckHIPError(hipStreamDestroy(state->stream));
  }
  memset(state, 0, sizeof(*state));
  free(state);
}

// Parses the additional_info argument. Returns 0 on error.
static int ParseAdditionalInfo(const char *arg, PluginState *state) {
  cJSON *root = NULL;
  cJSON *entry = NULL;
  int iterations = 0;
  int buffer_size = 0;
  root = cJSON_Parse(arg);
  if (!root) {
    printf("Invalid additional_info for random_walk\n");
    return 0;
  }
  entry = cJSON_GetObjectItem(root, "iterations");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Invalid \"iterations\" additional_info field for random_walk\n");
    cJSON_Delete(root);
    return 0;
  }
  iterations = entry->valueint;
  entry = cJSON_GetObjectItem(root, "buffer_size");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Invalid \"buffer_size\" additional_info field for random_walk\n");
    cJSON_Delete(root);
    return 0;
  }
  buffer_size = entry->valueint;
  cJSON_Delete(root);
  root = NULL;
  entry = NULL;
  if ((buffer_size <= 0) || (iterations <= 0)) {
    printf("Invalid settings for random_walk\n");
    return 0;
  }
  state->iterations = iterations;
  // Round the buffer size up to the next 64-bit aligned value.
  if ((buffer_size % sizeof(uint64_t)) != 0) {
    buffer_size += sizeof(uint64_t) - (buffer_size % sizeof(uint64_t));
  }
  state->buffer_length = buffer_size / sizeof(uint64_t);
  return 1;
}

// Allocates host and device memory for the plugin. Also initializes the
// random-walk buffer. Returns 0 on error.
static int AllocateMemory(PluginState *state) {
  uint64_t *host_random_buffer = NULL;
  uint64_t i;
  size_t buffer_size = (state->block_count * 2) * sizeof(uint64_t);

  // Allocate host and device memory for block times.
  if (!CheckHIPError(hipHostMalloc(&(state->kernel_times.block_times),
    buffer_size))) {
    return 0;
  }
  if (!CheckHIPError(hipMalloc(&(state->device_block_times), buffer_size))) {
    return 0;
  }

  // Initialize the device random-walk buffer. We first allocate the random
  // buffer host-side, then copy it to the device.
  buffer_size = state->buffer_length * sizeof(uint64_t);
  if (!CheckHIPError(hipMalloc(&(state->device_random_buffer), buffer_size))) {
    return 0;
  }
  if (!CheckHIPError(hipHostMalloc(&host_random_buffer, buffer_size))) {
    return 0;
  }
  for (i = 0; i < state->buffer_length; i++) {
    host_random_buffer[i] = i;
  }
  ShuffleArray(host_random_buffer, state->buffer_length);
  if (!CheckHIPError(hipMemcpy(state->device_random_buffer, host_random_buffer,
    buffer_size, hipMemcpyHostToDevice))) {
    hipHostFree(host_random_buffer);
    return 0;
  }
  hipHostFree(host_random_buffer);
  return 1;
}

// This kernel uses a single thread to access every element in the buffer,
// which should bring small buffers into the GPU cache. The accumulator can
// be NULL, and is used to prevent optimizations.
__global__ void InitialWalk(uint64_t *walk_buffer, uint64_t buffer_length,
    uint64_t *accumulator) {
  uint64_t i = 0;
  uint64_t result = 0;
  if (hipBlockIdx_x != 0) return;
  if (hipThreadIdx_x != 0) return;
  for (i = 0; i < buffer_length; i++) {
    result += walk_buffer[i];
  }
  if (accumulator != NULL) *accumulator = result;
}

static void* Initialize(InitializationParameters *params) {
  PluginState *state = NULL;
  if (!CheckHIPError(hipSetDevice(params->device_id))) return NULL;
  if (!CheckHIPError(hipSetDeviceFlags(PLUGIN_DEVICE_FLAGS))) return NULL;
  state = (PluginState *) calloc(1, sizeof(*state));
  if (!state) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }
  if (!GetSingleBlockAndGridDimensions(params, &state->thread_count,
    &state->block_count)) {
    Cleanup(state);
    return NULL;
  }
  if (!ParseAdditionalInfo(params->additional_info, state)) {
    Cleanup(state);
    return NULL;
  }
  if (!CheckHIPError(CreateHIPStreamWithMask(&(state->stream),
    params->compute_unit_mask, COMPUTE_UNIT_MASK_ENTRIES))) {
    Cleanup(state);
    return NULL;
  }
  state->stream_created = 1;
  state->kernel_times.kernel_name = "random_walk";
  state->kernel_times.thread_count = state->thread_count;
  state->kernel_times.block_count = state->block_count;
  state->kernel_times.shared_memory = 0;
  if (!AllocateMemory(state)) {
    Cleanup(state);
    return NULL;
  }
  hipLaunchKernelGGL(InitialWalk, 1, 1, 0, state->stream,
    state->device_random_buffer, state->buffer_length, (uint64_t *) NULL);
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) {
    Cleanup(state);
    return NULL;
  }
  return state;
}

// This initializes the block_times buffer on the GPU so that the algorithm
// for recording block times can work without synchronization.
static int CopyIn(void *data) {
  PluginState *state = (PluginState *) data;
  size_t times_size = state->block_count * 2 * sizeof(uint64_t);
  if (!CheckHIPError(hipMemsetAsync(state->device_block_times, 0xff,
    times_size, state->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  return 1;
}

// The kernel for randomly traversing the buffer. The dummy argument should be
// NULL, it exists to prevent optimizing out the loop.
__global__ void RandomWalkKernel(uint64_t iterations, uint64_t *accumulator,
  uint64_t *walk_buffer, uint64_t buffer_length, uint64_t *block_times) {
  uint64_t walk_index, i, result;
  uint64_t start_clock = clock64();
  if (start_clock < block_times[hipBlockIdx_x * 2]) {
    block_times[hipBlockIdx_x * 2] = start_clock;
  }
  walk_index = ((hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x) %
    buffer_length;
  result = 0;
  // The actual random-walk loop. Shuffling the array during initialization
  // makes a random cycle through the buffer.
  for (i = 0; i < iterations; i++) {
    walk_index = walk_buffer[walk_index];
    result += walk_index;
  }
  if (accumulator) *accumulator = result;
  block_times[hipBlockIdx_x * 2 + 1] = clock64();
}

static int Execute(void *data) {
  PluginState *state = (PluginState *) data;
  state->kernel_times.kernel_launch_times[0] = CurrentSeconds();
  hipLaunchKernelGGL(RandomWalkKernel, state->block_count,
    state->thread_count, 0, state->stream, state->iterations,
    (uint64_t *) NULL, state->device_random_buffer,
    state->buffer_length, state->device_block_times);
  state->kernel_times.kernel_launch_times[1] = CurrentSeconds();
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  state->kernel_times.kernel_launch_times[2] = CurrentSeconds();
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  PluginState *state = (PluginState *) data;
  size_t times_size = state->block_count * 2 * sizeof(uint64_t);
  if (!CheckHIPError(hipMemcpyAsync(state->kernel_times.block_times,
    state->device_block_times, times_size, hipMemcpyDeviceToHost,
    state->stream))) {
    return 0;
  }
  times->kernel_count = 1;
  times->kernel_times = &(state->kernel_times);
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  return 1;
}

static const char* GetName(void) {
  return "Random Walk";
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

