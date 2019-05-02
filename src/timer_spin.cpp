// This plugin spins on the GPU until a set number of clock ticks have elapsed.
// It takes a single additional_info parameter specifying the number of clock
// ticks.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hip/hip_runtime.h>
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
  // The number of clock ticks each thread must wait for.
  uint64_t ticks;
  uint64_t *device_block_times;
  // Holds times that are shared with the plugin host.
  KernelTimes kernel_times;
} PluginState;

// Implements the Cleanup() function required by the plugin interface.
static void Cleanup(void *data) {
  PluginState *state = (PluginState *) data;
  if (state->device_block_times) hipFree(state->device_block_times);
  if (state->kernel_times.block_times) {
    hipHostFree(state->kernel_times.block_times);
  }
  if (state->stream_created) {
    CheckHIPError(hipStreamDestroy(state->stream));
  }
  memset(state, 0, sizeof(*state));
  free(state);
}

// Parses the additional_info argument to get the clock ticks. Returns 0 on
// error.
static int SetClockTicks(const char *arg, PluginState *state) {
  int64_t parsed_value;
  char *end = NULL;
  if (!arg || (strlen(arg) == 0)) {
    printf("A number of clock ticks is required for timer_spin\n");
    return 0;
  }
  parsed_value = strtoll(arg, &end, 10);
  if ((*end != 0) || (parsed_value < 0)) {
    printf("Invalid clock tick count: %s\n", arg);
    return 0;
  }
  state->ticks = parsed_value;
  return 1;
}

// Allocates host and device memory for the plugin. Returns 0 on error.
static int AllocateMemory(PluginState *state) {
  size_t buffer_size = (state->block_count * 2) * sizeof(uint64_t);
  if (!CheckHIPError(hipHostMalloc(&(state->kernel_times.block_times),
    buffer_size))) {
    return 0;
  }
  if (!CheckHIPError(hipMalloc(&(state->device_block_times), buffer_size))) {
    return 0;
  }
  return 1;
}

static void* Initialize(InitializationParameters *params) {
  PluginState *state = NULL;
  state = (PluginState *) malloc(sizeof(*state));
  if (!state) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }
  memset(state, 0, sizeof(*state));
  if (!CheckHIPError(hipSetDevice(params->device_id))) {
    Cleanup(state);
    return NULL;
  }
  state->block_count = params->block_count;
  state->thread_count = params->thread_count;
  if (!SetClockTicks(params->additional_info, state)) {
    Cleanup(state);
    return NULL;
  }
  if (!CheckHIPError(CreateHIPStreamWithMask(&(state->stream),
    params->compute_unit_mask, COMPUTE_UNIT_MASK_ENTRIES))) {
    Cleanup(state);
    return NULL;
  }
  state->stream_created = 1;
  state->kernel_times.kernel_name = "timer_spin";
  state->kernel_times.thread_count = state->thread_count;
  state->kernel_times.block_count = state->block_count;
  state->kernel_times.shared_memory = 0;
  if (!AllocateMemory(state)) {
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

// The kernel for spinning until a certain number of clock ticks have elapsed.
// The dummy argument should be NULL, it exists to prevent optimizing out the
// loop.
__global__ void TimerSpinKernel(uint64_t ticks, uint64_t *dummy,
  uint64_t *block_times) {
  uint64_t accumulator;
  uint64_t start_clock = clock64();
  if (start_clock < block_times[hipBlockIdx_x * 2]) {
    block_times[hipBlockIdx_x * 2] = start_clock;
  }
  accumulator = 0;
  while ((clock64() - start_clock) < ticks) {
    continue;
  }
  /*
  current_clock = clock64();
  while ((current_clock - start_clock) < ticks) {
    accumulator += current_clock % hipBlockIdx_x;
    current_clock = clock64();
  }
  */
  if (dummy) *dummy = accumulator;
  block_times[hipBlockIdx_x * 2 + 1] = clock64();
}

static int Execute(void *data) {
  PluginState *state = (PluginState *) data;
  state->kernel_times.kernel_launch_times[0] = CurrentSeconds();
  printf("Starting to spin for %llu ticks.\n", (unsigned long long) state->ticks);
  hipLaunchKernelGGL(TimerSpinKernel, state->block_count,
    state->thread_count, 0, state->stream, state->ticks,
    (uint64_t *) NULL, state->device_block_times);
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
  return "Timer Spin";
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

