// This plugin is intended to put artificial pressure on the number of hardware
// queues supported by AMD GPUs. It does this by creating a given number of
// streams.  If this is done in combination with the use_cu_mask setting being
// "true", then every stream should be backed by a separate HSA queue (at least
// under the current ROCclr implementation).  If use_cu_mask is false, then any
// number of streams will be share a number of HSA queues specified by the
// GPU_MAX_HW_QUEUES, defined in ROCclr/utils/flags.hpp (defaults to 4).
// This plugin launches a simple kernel into each stream during its Execute
// phase to guarantee that the streams are actually created and associated with
// some queue. It respects the thread_count and block_count settings, but the
// kernel's behavior shouldn't change with more threads or blocks.
//
// This plugin's additional_info must be a JSON object with the following keys:
//
// - "stream_count": The number of HIP streams to create. Required. For
//   convenience, this may be zero, in which case the entire plugin does
//   nothing.
//
// - "use_cu_mask": A boolean, defaulting to false. If true, the plugin will
//   create each stream using the cu_mask setting specified in the plugin's
//   config.  If false, each stream will be created without a CU mask.

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
  // The list of hipStream_t instances created by this plugin.
  hipStream_t *streams;
  // This is a pointer to a single uint64_t on the GPU, which we use to launch
  // a bunch of useless but tiny memory copies.
  uint64_t *dummy_buffer_d;
  // A copy of the dummy buffer on the host.
  uint64_t dummy_buffer_h;
  // The number of streams in the list.
  int stream_count;
  // The number of threads and blocks to launch.
  int thread_count, block_count;
  // Will be 1 if the "use_cu_mask" setting was true.
  int use_cu_mask;
} PluginState;

static void Cleanup(void *data) {
  int i;
  PluginState *state = (PluginState *) data;
  if (!state) return;
  CheckHIPError(hipFree(state->dummy_buffer_d));
  for (i = 0; i < state->stream_count; i++) {
    if (state->streams[i]) {
      CheckHIPError(hipStreamDestroy(state->streams[i]));
    }
  }
  free(state->streams);
  memset(state, 0, sizeof(*state));
  free(state);
}

// Parses the additional_info argument. Returns 0 on error.
static int ParseAdditionalInfo(const char *arg, PluginState *state) {
  cJSON *root = NULL;
  cJSON *entry = NULL;
  root = cJSON_Parse(arg);
  if (!root) {
    printf("Invalid additional_info for dummy_streams.\n");
    return 0;
  }

  // Make sure that vector_length is present and positive.
  entry = cJSON_GetObjectItem(root, "stream_count");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Invalid stream_count setting.\n");
    cJSON_Delete(root);
    return 0;
  }
  if (entry->valueint < 0) {
    printf("The stream_count setting can't be negative.\n");
    cJSON_Delete(root);
    return 0;
  }
  state->stream_count = entry->valueint;

  // Check the use_cu_mask setting, if it's present.
  entry = cJSON_GetObjectItem(root, "use_cu_mask");
  if (entry) {
    if ((entry->type != cJSON_False) && (entry->type != cJSON_True)) {
      printf("The use_cu_mask setting must be a boolean.\n");
      cJSON_Delete(root);
      return 0;
    }
    state->use_cu_mask = entry->type == cJSON_True;
  }

  cJSON_Delete(root);
  root = NULL;
  entry = NULL;
  return 1;
}

// Creates the streams. Assumes state->streams has already been allocated, and
// additional info has been parsed. Returns 0 on error.
static int CreateStreams(InitializationParameters *params,
    PluginState *state) {
  int i;
  for (i = 0; i < state->stream_count; i++) {
    if (!state->use_cu_mask) {
      if (!CheckHIPError(hipStreamCreate(state->streams + i))) {
        return 0;
      }
    } else {
      if (!CheckHIPError(hipExtStreamCreateWithCUMask(state->streams + i,
        COMPUTE_UNIT_MASK_ENTRIES, params->compute_unit_mask))) {
        return 0;
      }
    }
  }
  return 1;
}

static void* Initialize(InitializationParameters *params) {
#ifndef __HIP__
  printf("dummy_streams requires using AMD GPUs supporting CU masking.\n");
  return NULL;
#else
  PluginState *state = NULL;
  if (!CheckHIPError(hipSetDevice(params->device_id))) return NULL;
  if (!CheckHIPError(hipSetDeviceFlags(PLUGIN_DEVICE_FLAGS))) return NULL;
  state = (PluginState *) calloc(sizeof(*state), 1);
  if (!state) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }
  state->thread_count = params->thread_count;
  state->block_count = params->block_count;

  // Parse the additional info, and allocate space to hold the stream handles.
  if (!ParseAdditionalInfo(params->additional_info, state)) {
    Cleanup(state);
    return NULL;
  }
  state->streams = (hipStream_t *) calloc(state->stream_count,
    sizeof(hipStream_t));
  if (!state->streams) {
    printf("Failed allocating stream handles.\n");
    free(state);
    return NULL;
  }
  if (!CheckHIPError(hipMalloc(&state->dummy_buffer_d, sizeof(uint64_t)))) {
    printf("Failed allocating dummy buffer for memcpy's.\n");
    Cleanup(state);
    return NULL;
  }

  // Finally, create the streams.
  if (!CreateStreams(params, state)) {
    printf("Failed creating streams.\n");
    Cleanup(state);
    return NULL;
  }
  return state;
#endif  // HIP-only stuff
}

// We don't need to copy anything in.
static int CopyIn(void *data) {
  return 1;
}

// Does basically nothing except write to the buffer.
__global__ void DummyWorkKernel(uint64_t *buffer) {
  uint64_t v = blockIdx.x * blockDim.x + threadIdx.x;
  *buffer = v;
}

// The execute function just does a bit of work to use the streams. For now,
// it launches a minimal kernel in each stream, then synchronizes with all of
// them.
//
// TODO: Record block and kernel launch times for each of the dummy kernels.
static int Execute(void *data) {
  PluginState *state = (PluginState *) data;
  int i;
  for (i = 0; i < state->stream_count; i++) {
    hipLaunchKernelGGL(DummyWorkKernel, state->block_count,
      state->thread_count, 0, state->streams[i], state->dummy_buffer_d);
  }
  for (i = 0; i < state->stream_count; i++) {
    if (!CheckHIPError(hipStreamSynchronize(state->streams[i]))) return 0;
  }
  return 1;
}

// No need to copy anything out, but just let the framework know that we ran no
// kernels.
static int CopyOut(void *data, TimingInformation *times) {
  PluginState *state = (PluginState *) data;
  times->kernel_count = 0;
  times->kernel_times = NULL;
  times->resulting_data_size = sizeof(uint64_t);
  times->resulting_data = &state->dummy_buffer_h;
  // We ran no GPU work if our stream_count was 0.
  if (state->stream_count == 0) return 1;

  // We'll just use stream 0 as an arbitrary choice for copying the dummy data
  // back to the host.
  if (!CheckHIPError(hipMemcpyAsync(&(state->dummy_buffer_h),
    state->dummy_buffer_d, sizeof(uint64_t), hipMemcpyDeviceToHost,
    state->streams[0]))) {
    return 0;
  }
  if (!CheckHIPError(hipStreamSynchronize(state->streams[0]))) return 0;
  return 1;
}

static const char* GetName(void) {
  return "Dummy Streams";
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

