// This plugin is intended to put artificial pressure on the number of hardware
// queues supported by AMD GPUs. It does this by creating a number of HIP
// streams with custom CU masks, which ROCclr backs using an entirely new HSA
// queue. (In the future, if ROCclr gets "smarter" about CU masks in HSA
// queues, this may need to change. However, it works with ROCm 3.7.) It calls
// hipStreamSynchronize once per stream, during each "execute" phase, but
// otherwise launches no work. This requires an AMD GPU supporting CU masking.
//
// This plugin's additional_info must be a JSON object with the following keys:
//
// - "stream_count": The number of HIP streams to create. Required. For
//   convenience, this may be zero, in which case the entire plugin does
//   nothing.

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
  // The number of streams in the list.
  int stream_count;
} PluginState;

static void Cleanup(void *data) {
  int i;
  PluginState *state = (PluginState *) data;
  if (!state) return;
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

  cJSON_Delete(root);
  root = NULL;
  entry = NULL;
  return 1;
}

static void* Initialize(InitializationParameters *params) {
#ifndef __HIP__
  printf("dummy_streams requires using AMD GPUs supporting CU masking.\n");
  return NULL;
#else
  PluginState *state = NULL;
  uint32_t cu_mask;
  int i;
  if (!CheckHIPError(hipSetDevice(params->device_id))) return NULL;
  if (!CheckHIPError(hipSetDeviceFlags(PLUGIN_DEVICE_FLAGS))) return NULL;
  state = (PluginState *) calloc(sizeof(*state), 1);
  if (!state) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }

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

  // Finally, create the streams.
  cu_mask = 0xdeadbeef;
  for (i = 0; i < state->stream_count; i++) {
    if (!CheckHIPError(hipExtStreamCreateWithCUMask(state->streams + i,
      1, &cu_mask))) {
      printf("Failed creating stream %d.\n", i);
      Cleanup(state);
      return NULL;
    }
  }
  return state;
#endif  // HIP-only stuff
}

// We don't need to copy anything in.
static int CopyIn(void *data) {
  return 1;
}

// The execute function just does some dummy work here--sync with every stream.
// We never launch any work on the stream, and I haven't checked internally if
// this is necessary. Still, I'm trying to avoid the system ignoring our
// streams entirely.
//
// TODO: Test this. If it's not necessary, then make Execute(..) do nothing.
static int Execute(void *data) {
  PluginState *state = (PluginState *) data;
  int i;
  for (i = 0; i < state->stream_count; i++) {
    if (!CheckHIPError(hipStreamSynchronize(state->streams[i]))) return 0;
  }
  return 1;
}

// No need to copy anything out, but just let the framework know that we ran no
// kernels.
static int CopyOut(void *data, TimingInformation *times) {
  times->kernel_count = 0;
  times->kernel_times = NULL;
  times->resulting_data_size = 0;
  times->resulting_data = NULL;
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

