// This plugin only does a nominally small amount of GPU work in the "execute"
// phase, and instead is focused mostly on the time needed to copy data to GPU
// memory. The "additional_info" field contains a JSON object specifying the
// following fields:
//
// - "buffer_size" in bytes, the total size to copy
// - "copy_subdivisions", which specifies the number of "chunks" to divide the
//   buffer into. This can just be 1 to copy everything in one chunk. This is
//   the number of memory copies that will be issued. If buffer_size isn't
//   evenly divisible by this, it will be rounded up to the nearest multiple.
// - "sync_every_copy": A boolean. If true, then a hipStreamSynchronize will be
//   issued after every memory-copy of each chunk.
//
// In short, this plugin is intended to benchmark memory-copy performance and
// behavior.

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
  // The size of the memory to copy, in bytes.
  uint64_t buffer_size;
  // The maximum size of a "chunk" to copy. The last chunk copied may be
  // smaller than this, if buffer_size isn't evenly divisible by
  // copy_subdivisions.
  uint64_t copy_chunk_size;
  uint8_t *device_buffer;
  // The copy of the buffer on the host. Will be randomly initialized.
  uint8_t *host_buffer;
  uint64_t *device_block_times;
  // If nonzero, then a hipStreamSynchronize will be issued after every memory
  // copy.
  int sync_every_copy;
  // Holds times that are shared with the plugin host.
  KernelTimes kernel_times;
} PluginState;

// Returns a single random 64-bit value.
static uint64_t Random64(void) {
  // This won't fill in all bits, but we'll do it anyway to keep things faster.
  return (uint64_t) rand();
}

// Implements the Cleanup() function required by the plugin interface.
static void Cleanup(void *data) {
  PluginState *state = (PluginState *) data;
  if (!state) return;
  hipFree(state->device_block_times);
  hipHostFree(state->kernel_times.block_times);
  hipFree(state->device_buffer);
  hipHostFree(state->host_buffer);
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
  uint64_t buffer_size = 0;
  int copy_subdivisions = 0;
  root = cJSON_Parse(arg);
  if (!root) {
    printf("Invalid additional_info for memory_copy\n");
    return 0;
  }

  entry = cJSON_GetObjectItem(root, "copy_subdivisions");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Invalid or missing \"copy_subdivisions\" for memory_copy\n");
    cJSON_Delete(root);
    return 0;
  }
  copy_subdivisions = entry->valueint;
  if (copy_subdivisions <= 0) {
    printf("The \"copy_subdivisions\" for memory_copy must be positive.\n");
    cJSON_Delete(root);
    return 0;
  }

  entry = cJSON_GetObjectItem(root, "buffer_size");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Invalid \"buffer_size\" for memory_copy\n");
    cJSON_Delete(root);
    return 0;
  }
  buffer_size = (uint64_t) entry->valuedouble;

  entry = cJSON_GetObjectItem(root, "sync_every_copy");
  // If the entry is present, then it must be either cJSON_True or cJSON_False.
  if (entry) {
    if (entry->type == cJSON_True) {
      state->sync_every_copy = 1;
    } else if (entry->type != cJSON_False) {
      printf("Invalid sync_every_copy setting for memory_copy\n");
      cJSON_Delete(root);
      return 0;
    }
  }

  // Clean up the cJSON stuff and do some final sanity checks.
  cJSON_Delete(root);
  root = NULL;
  entry = NULL;
  if (buffer_size < copy_subdivisions) {
    printf("In memory_copy, \"buffer_size\" must be at least equal to "
      "\"copy_subdivisions\".\n");
    return 0;
  }

  // Round up the buffer size to a multiple of copy_subdivisions.
  if ((buffer_size % copy_subdivisions) != 0) {
    buffer_size += copy_subdivisions - (buffer_size % copy_subdivisions);
  }
  state->buffer_size = buffer_size;
  state->copy_chunk_size = buffer_size / copy_subdivisions;

  return 1;
}

// Allocates host and device memory for the plugin. Returns 0 on error.
static int AllocateMemory(PluginState *state) {
  uint64_t i, count;
  double tmp_time;
  size_t buffer_size = (state->block_count * 2) * sizeof(uint64_t);

  // Allocate host and device memory for block times.
  if (!CheckHIPError(hipHostMalloc(&(state->kernel_times.block_times),
    buffer_size))) {
    return 0;
  }
  if (!CheckHIPError(hipMalloc(&(state->device_block_times), buffer_size))) {
    return 0;
  }

  // Initialize the memory buffers, and randomly fill the host buffer.
  buffer_size = state->buffer_size;

  tmp_time = CurrentSeconds();
  if (!CheckHIPError(hipMalloc(&(state->device_buffer), buffer_size))) {
    return 0;
  }
  tmp_time = CurrentSeconds() - tmp_time;
  printf("Time for %f MB hipMalloc: %f seconds\n",
    ((double) buffer_size) / (1024.0 * 1024.0), tmp_time);

  tmp_time = CurrentSeconds();
  if (!CheckHIPError(hipHostMalloc(&state->host_buffer, buffer_size))) {
    return 0;
  }
  tmp_time = CurrentSeconds() - tmp_time;
  printf("Time for %f MB hipHostMalloc: %f seconds\n",
    ((double) buffer_size) / (1024.0 * 1024.0), tmp_time);

  count = buffer_size / sizeof(uint64_t);
  tmp_time = CurrentSeconds();
  for (i = 0; i < count; i++) {
    ((uint64_t *) state->host_buffer)[i] = Random64();
  }
  tmp_time = CurrentSeconds() - tmp_time;
  printf("Time for randomly filling buffer: %f seconds.\n", tmp_time);

  return 1;
}

static void* Initialize(InitializationParameters *params) {
  PluginState *state = NULL;
  if (!CheckHIPError(hipSetDevice(params->device_id))) return NULL;
  if (!CheckHIPError(hipSetDeviceFlags(hipDeviceScheduleYield))) return NULL;
  state = (PluginState *) malloc(sizeof(*state));
  if (!state) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }
  memset(state, 0, sizeof(*state));
  state->block_count = params->block_count;
  state->thread_count = params->thread_count;
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
  state->kernel_times.kernel_name = "memory_copy";
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
  size_t total_copied = 0;
  size_t times_size = state->block_count * 2 * sizeof(uint64_t);
  uint8_t *device_chunk = NULL;
  uint8_t *host_chunk = NULL;

  // First, reset the buffer for block times.
  if (!CheckHIPError(hipMemsetAsync(state->device_block_times, 0xff,
    times_size, state->stream))) {
    return 0;
  }

  // Next, issue the memory copies.
  while (total_copied < state->buffer_size) {
    device_chunk = state->device_buffer + total_copied;
    host_chunk = state->host_buffer + total_copied;
    if (!CheckHIPError(hipMemcpyAsync(device_chunk, host_chunk,
      state->copy_chunk_size, hipMemcpyHostToDevice, state->stream))) {
      return 0;
    }
    if (state->sync_every_copy) {
      if (!CheckHIPError(hipStreamSynchronize(state->stream))) {
        return 0;
      }
    }
    total_copied += state->copy_chunk_size;
  }

  // We always do at least one hipStreamsynchronize after all of the copies.
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  return 1;
}

// The kernel for doing some nominal work on the buffer (to prevent optimizing
// it out). Basically, shifts all of the entries up one index.
__global__ void ManipulateMemoryKernel(uint8_t *buffer, uint64_t buffer_size,
    uint64_t *block_times) {
  uint64_t start_clock = clock64();
  uint64_t *buffer64 = (uint64_t *) buffer;
  uint64_t length64 = buffer_size / sizeof(uint64_t);
  uint64_t index;
  if (start_clock < block_times[blockIdx.x * 2]) {
    block_times[blockIdx.x * 2] = start_clock;
  }
  // We'll treat the buffer as an array of 64-bit values.
  index = blockIdx.x * blockDim.x + threadIdx.x;

  // Shift the value up one position. We don't care about thread safety for
  // this; it's an arbitrary dummy operation.
  if (index < (length64 - 1)) {
    buffer64[index + 1] = buffer64[index];
  }

  block_times[blockIdx.x * 2 + 1] = clock64();
}

static int Execute(void *data) {
  PluginState *state = (PluginState *) data;

  state->kernel_times.kernel_launch_times[0] = CurrentSeconds();
  hipLaunchKernelGGL(ManipulateMemoryKernel, state->block_count,
    state->thread_count, 0, state->stream, state->device_buffer,
    state->buffer_size, state->device_block_times);
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
  return "Memory Copy";
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

