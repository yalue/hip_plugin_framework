// This plugin allows submitting potentially many kernels and memory-transfer
// operations to a single stream during a single "Execute" phase. While useful
// for mimicing more complex systems, this has the unfortunate side effect of
// making it much more complex to configure.  The plugin is configured almost
// entirely using the "additional_info" config field, which takes the form of
// a JSON array:
// "additional_info": [
//   {
//     "action": "<A string, must be one of 'kernel', 'copy_in', 'copy_out', or
//       'synchronize'.>",
//     "matrix_width": <A number, used only for the 'kernel' action. The kernel
//       carries out a matrix multiplication with the given width.>,
//     "block_dims": <Must be an array with two numbers. Only used with the
//       'kernel' action.>,
//     "size": <A number of bytes. Used with the 'copy_in' and 'copy_out'
//       actions. Will be rounded up to hold an array of floats.>
//   },
//   { ... <More actions here> ...}
// ]
//
// Note that a hipStreamSynchronize will occur at the end of the Execute phase,
// regardless of the presence or lack of a "synchronize" action at the end of
// the list.
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include "third_party/cJSON.h"
#include "plugin_hip_utilities.h"
#include "plugin_interface.h"
#include "plugin_utilities.h"

// The types of operations to carry out.
typedef enum {
  KERNEL_OPERATION,
  COPY_IN_OPERATION,
  COPY_OUT_OPERATION,
  SYNCHRONIZE_OPERATION,
  INVALID_OPERATION,
} ActionType;

// Contains info about each action that is carried out.
typedef struct {
  ActionType type;
  // The width of the square matrices to multiply.
  int matrix_width;
  // The number of bytes needed by the copy, or the total number of bytes
  // needed for the matrix multiplication (3 x matrix_width * matrix_width *
  // sizeof(float)), for 2 source matrices and 1 result matrix.
  size_t data_size;
  // The number of threads for the matrix multiply, specified in the config.
  dim3 block_size;
  // The number of blocks needed for matrix-multiply launches.
  dim3 grid_size;
  char *label;
} ActionInfo;

// Holds the state of an instance of this plugin.
typedef struct {
  hipStream_t stream;
  int stream_created;

  // The list of actions to carry out, and the number of actions. kernel_count
  // is the number of actions that are kernel launches.
  ActionInfo *actions;
  int action_count;
  int kernel_count;

  // Pointers to the host and device memory. Will be equal in size to the
  // largest memory-transfer size, or the largest matrix-multiply size.
  size_t buffer_size;
  float *device_memory;
  float *host_memory;

  // A list of buffers to hold kernel block times on the device. (Pointers to
  // host-side copies are in the kernel_times list.) Ordered by the order in
  // which kernels are launched; contains kernel_count entries.
  uint64_t **device_block_times;

  // Contains a list of per-kernel timing data required by the plugin
  // framework. Contains kernel_count entries.
  KernelTimes *kernel_times;
} PluginState;

// Implements the Cleanup() function required by the plugin interface.
static void Cleanup(void *data) {
  int i;
  PluginState *state = (PluginState *) data;
  hipFree(state->device_memory);
  hipHostFree(state->host_memory);
  for (i = 0; i < state->kernel_count; i++) {
    // A bunch of these per-kernel pointers can only be freed if we've
    // successfully allocated the lists that hold them.
    if (state->device_block_times) {
      hipFree(state->device_block_times[i]);
    }
    if (state->kernel_times) {
      hipHostFree(state->kernel_times[i].block_times);
    }
  }
  for (i = 0; i < state->action_count; i++) {
    free(state->actions[i].label);
  }
  free(state->device_block_times);
  free(state->actions);
  if (state->stream_created) {
    CheckHIPError(hipStreamDestroy(state->stream));
  }
  memset(state, 0, sizeof(*state));
  free(state);
}

// Parses the cJSON object, expected to be a string, containing the type of
// operation to carry out. Returns INVALID_OPERATION if anything's invalid.
static ActionType ParseActionType(cJSON *entry) {
  const char *v;
  if (!entry || (entry->type != cJSON_String)) {
    printf("Each stream_actions \"action\" setting must be present, and a "
      "string.\n");
    return INVALID_OPERATION;
  }
  v = entry->valuestring;
  if (strcmp(v, "kernel") == 0) return KERNEL_OPERATION;
  if (strcmp(v, "copy_in") == 0) return COPY_IN_OPERATION;
  if (strcmp(v, "copy_out") == 0) return COPY_OUT_OPERATION;
  if (strcmp(v, "synchronize") == 0) return SYNCHRONIZE_OPERATION;
  printf("Invalid \"action\" setting for stream_actions: %s\n", v);
  return INVALID_OPERATION;
}

// Parses the given JSON object, which must be an array of numbers with *two*
// entries. The result dim3 will always have its third entry set to 1. Returns
// 0 on error.
static int ParseJSONDims(cJSON *obj, dim3 *result) {
  cJSON *child;
  int values[2];
  int i;
  if (!obj || (obj->type != cJSON_Array)) {
    printf("Didn't get block_dims JSON array.\n");
    return 0;
  }
  child = obj->child;
  for (i = 0; i < 2; i++) {
    if (!child || (child->type != cJSON_Number)) {
      printf("block_dims didn't contain enough numbers.\n");
      return 0;
    }
    if (child->valueint <= 0) {
      printf("Block dimensions must be positive.\n");
      return 0;
    }
    values[i] = child->valueint;
    child = child->next;
  }
  // If there's still a child, then child->next was non-NULL on the second
  // entry, indicating too many entries.
  if (child) {
    printf("Got more than two block dimensions, currently unsupported for "
      "stream_action.\n");
    return 0;
  }
  result->x = values[0];
  result->y = values[1];
  result->z = 1;
  return 1;
}

// Parses kernel-launch-related data from the given JSON object, writing it
// to the info struct. Returns 0 on error.
static int ParseKernelAction(ActionInfo *info, cJSON *obj) {
  cJSON *entry = NULL;
  int blocks_wide, blocks_tall, matrix_width;
  entry = cJSON_GetObjectItem(obj, "block_dims");
  if (!ParseJSONDims(entry, &(info->block_size))) return 0;
  entry = cJSON_GetObjectItem(obj, "matrix_width");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Invalid or missing matrix_width setting.\n");
    return 0;
  }
  if (entry->valueint <= 0) {
    printf("The matrix_width setting must be positive.\n");
    return 0;
  }
  matrix_width = entry->valueint;
  info->matrix_width = matrix_width;
  blocks_wide = matrix_width / info->block_size.x;
  if ((matrix_width % info->block_size.x) != 0) blocks_wide++;
  blocks_tall = matrix_width / info->block_size.y;
  if ((matrix_width % info->block_size.y) != 0) blocks_tall++;
  info->grid_size.x = blocks_wide;
  info->grid_size.y = blocks_tall;
  info->grid_size.z = 1;

  // We need enough bytes for 3 matrices: two input and one output.
  info->data_size = matrix_width * matrix_width * sizeof(float) * 3;
  return 1;
}

// Parses copy-related data from the given JSON object, writing it to the info
// struct. Returns 0 on error.
static int ParseCopyAction(ActionInfo *info, cJSON *obj) {
  cJSON *entry = NULL;
  entry = cJSON_GetObjectItem(obj, "size");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Invalid or missing size setting for copy operation.\n");
    return 0;
  }
  if (entry->valuedouble < 0) {
    printf("The size of a copy operation must be positive.\n");
    return 0;
  }
  info->data_size = entry->valuedouble;
  // Make sure the size is rounded up to hold a whole number of floats.
  while ((info->data_size % sizeof(float)) != 0) info->data_size++;
  return 1;
}

// Parses the additional_info in order to allocate and populate state->actions.
// Sets action_count and kernel_count, but does not allocate any other memory
// or validate any settings apart from their presence in the config. Returns 0
// on error.
static int ParseAdditionalInfo(const char *arg, PluginState *state) {
  cJSON *root = NULL;
  cJSON *action_json = NULL;
  cJSON *entry = NULL;
  int i, entry_count = 0;
  ActionType action_type;
  if (!arg || (strlen(arg) == 0)) {
    printf("Missing additional_info for stream_actions.\n");
    return 0;
  }
  root = cJSON_Parse(arg);
  if (!root) {
    printf("Invalid additional_info for stream_actions.\n");
    return 0;
  }
  if (root->type != cJSON_Array) {
    printf("additional_info for stream_actions must be a JSON object.\n");
    cJSON_Delete(root);
    return 0;
  }
  action_json = root->child;
  if (!action_json) {
    cJSON_Delete(root);
    printf("stream_actions must be configured with at least one action.\n");
    return 0;
  }

  // Count the actions in the list so we can allocate our internal ActionInfo
  // list. Also verify that everything's a cJSON object.
  entry_count = 1;
  action_json = action_json->next;
  while (action_json) {
    if (action_json->type != cJSON_Object) {
      cJSON_Delete(root);
      printf("Found non-object in stream_actions' additional_info array.\n");
      return 0;
    }
    action_json = action_json->next;
    entry_count++;
  }
  state->actions = (ActionInfo *) calloc(entry_count, sizeof(ActionInfo));
  if (!state->actions) {
    cJSON_Delete(root);
    printf("Failed allocating memory for the list of actions.\n");
    return 0;
  }

  // Now we can go through the list again and actually obtain the data.
  action_json = root->child;
  for (i = 0; i < entry_count; i++) {
    entry = cJSON_GetObjectItem(action_json, "action");
    action_type = ParseActionType(entry);
    if (action_type == INVALID_OPERATION) {
      cJSON_Delete(root);
      return 0;
    }
    state->action_count++;
    state->actions[i].type = action_type;
    if (action_type == KERNEL_OPERATION) {
      if (!ParseKernelAction(state->actions + i, action_json)) {
        cJSON_Delete(root);
        return 0;
      }
      state->kernel_count++;
    }
    if ((action_type == COPY_IN_OPERATION) ||
      (action_type == COPY_OUT_OPERATION)) {
      if (!ParseCopyAction(state->actions + i, action_json)) {
        cJSON_Delete(root);
        return 0;
      }
    }
    // No additional parsing is needed for synchronize operations.
    action_json = action_json->next;
  }

  cJSON_Delete(root);
  return 1;
}

// Formats a string using printf syntax, returning a newly allocated char*
// containing the formatted string. The returned string must be freed when no
// longer needed. Returns NULL on error.
static char* AllocatePrintf(const char *fmt, ...) {
  // Abitrary limit to 256 bytes, plenty for our application.
  char buffer[256];
  va_list args;
  memset(buffer, 0, sizeof(buffer));
  va_start(args, fmt);
  vsnprintf(buffer, sizeof(buffer) - 1, fmt, args);
  va_end(args);
  return strdup(buffer);
}

static float RandomFloat(void) {
  return ((float) rand()) / ((float) RAND_MAX);
}

static int Dim3Size(dim3 v) {
  return v.x * v.y * v.z;
}

// Allocates memory, including GPU memory, needed to support all of the
// operations specified in the config. Expects state->stream to have already
// been created. Returns 0 on error.
static int AllocateMemory(PluginState *state) {
  size_t i, max_size = 0;
  int kernel_number;
  KernelTimes *tmp = NULL;

  // First, determine the maximum size needed by any of the operations so we
  // can allocate and reuse a single buffer for all of them.
  for (i = 0; i < state->action_count; i++) {
    if (state->actions[i].data_size > max_size) {
      max_size = state->actions[i].data_size;
    }
  }
  state->buffer_size = max_size;
  if (!CheckHIPError(hipMalloc(&(state->device_memory), max_size))) return 0;
  if (!CheckHIPError(hipHostMalloc(&(state->host_memory), max_size))) return 0;

  // Randomly initialize the host and device data buffers.
  for (i = 0; i < (max_size / sizeof(float)); i++) {
    state->host_memory[i] = RandomFloat();
  }
  if (!CheckHIPError(hipMemcpyAsync(state->device_memory, state->host_memory,
    max_size, hipMemcpyHostToDevice, state->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;

  // Allocate the per-kernel bookkeeping data.
  state->device_block_times = (uint64_t **) calloc(state->kernel_count,
    sizeof(uint64_t*));
  if (!state->device_block_times) {
    printf("Failed allocating list of block-times pointers.\n");
    return 0;
  }
  state->kernel_times = (KernelTimes *) calloc(state->kernel_count,
    sizeof(KernelTimes));
  if (!state->kernel_times) {
    printf("Failed allocating list of kernel-times structs.\n");
    return 0;
  }
  kernel_number = 0;
  for (i = 0; i < state->action_count; i++) {
    if (state->actions[i].type != KERNEL_OPERATION) continue;
    tmp = state->kernel_times + kernel_number;
    tmp->block_count = Dim3Size(state->actions[i].grid_size);
    tmp->thread_count = Dim3Size(state->actions[i].block_size);

    // For each kernel, we need to allocate memory to hold the name and block
    // times.
    state->actions[i].label = AllocatePrintf("Kernel #%d", kernel_number);
    if (!state->actions[i].label) {
      printf("Failed allocating kernel %d name.\n", kernel_number);
      return 0;
    }
    tmp->kernel_name = state->actions[i].label;
    if (!CheckHIPError(hipHostMalloc(&(tmp->block_times), tmp->block_count *
      2 * sizeof(uint64_t)))) {
      return 0;
    }
    if (!CheckHIPError(hipMalloc(&(state->device_block_times[kernel_number]),
      tmp->block_count * 2 * sizeof(uint64_t)))) {
      return 0;
    }
    kernel_number++;
  }

  return 1;
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
  if (!ParseAdditionalInfo(params->additional_info, state)) {
    Cleanup(state);
    return NULL;
  }
  // We use the stream for initializing device memory, so create it before
  // AllocateMemory.
  if (!CheckHIPError(CreateHIPStreamWithMask(&(state->stream),
    params->compute_unit_mask, COMPUTE_UNIT_MASK_ENTRIES))) {
    Cleanup(state);
    return NULL;
  }
  state->stream_created = 1;
  if (!AllocateMemory(state)) {
    Cleanup(state);
    return NULL;
  }
  return state;
}

static int CopyIn(void *data) {
  PluginState *state = (PluginState *) data;
  size_t times_size;
  int i;
  // We don't actually do the copy actions in this, since they all need to be
  // done during the Execute phase. However, we still use this opportunity to
  // reset our block-time measurements.
  for (i = 0; i < state->kernel_count; i++) {
    times_size = state->kernel_times[i].block_count * 2 * sizeof(uint64_t);
    if (!CheckHIPError(hipMemsetAsync(state->device_block_times[i], 0xff,
      times_size, state->stream))) {
      return 0;
    }
  }
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  return 1;
}

// The GPU kernel for carrying out matrix multiplication. Expects a 2D grid
// with sufficient threads to cover the entire matrix.
__global__ void MatrixMultiplyKernel(float *a, float *b, float *c, int width,
  uint64_t *block_times) {
  int row, col, k, block_index;
  float v_a, v_b, v_c;
  uint64_t start_clock = clock64();
  block_index = blockIdx.y * gridDim.x + blockIdx.x;
  if (start_clock < block_times[block_index * 2]) {
    block_times[block_index * 2] = start_clock;
  }

  // The row and column of the element in the output matrix is determined by
  // the thread's position in the 2D grid.
  col = blockIdx.x * blockDim.x + threadIdx.x;
  row = blockIdx.y * blockDim.y + threadIdx.y;
  if ((col >= width) || (row >= width)) {
    // Don't try doing computations if we're outside of the matrix.
    block_times[block_index * 2 + 1] = clock64();
    return;
  }

  // Actually carry out the multiplication for this thread's element.
  v_c = 0;
  for (k = 0; k < width; k++) {
    v_a = a[row * width + k];
    v_b = b[k * width + col];
    v_c += v_a * v_b;
  }
  c[row * width + col] = v_c;
  block_times[block_index * 2 + 1] = clock64();
}

// Launches the kernel corresponding to action i. kernel_number is the index
// into the kernel_times and device_block_times arrays for the kernel. Returns
// 0 on error.
static int LaunchKernelAction(PluginState *state, int i, int kernel_number) {
  KernelTimes *k = state->kernel_times + kernel_number;
  ActionInfo *action = state->actions + i;
  uint64_t *b = state->device_block_times[kernel_number];
  int matrix_width = action->matrix_width;
  float *m_a, *m_b, *m_c;
  // Get the pointers to the source and result matrices. We just arbitrarily
  // put the three matrices sequentially in the device memory.
  m_a = state->device_memory;
  m_b = m_a + (matrix_width * matrix_width);
  m_c = m_b + (matrix_width * matrix_width);

  k->kernel_launch_times[0] = CurrentSeconds();
  hipLaunchKernelGGL(MatrixMultiplyKernel, action->grid_size,
    action->block_size, 0, state->stream, m_a, m_b, m_c, matrix_width, b);
  k->kernel_launch_times[1] = CurrentSeconds();

  // We unfortunately can't record the CPU time at which this kernel completes,
  // since we don't want to do a stream synchronize here.
  return 1;
}

// Launches action i, which is assumed to be a memory-copy action. Returns 0 on
// error.
static int LaunchCopyAction(PluginState *state, int i) {
  ActionInfo *action = state->actions + i;
  hipMemcpyKind direction;
  float *src = NULL, *dst = NULL;
  if (action->type == COPY_IN_OPERATION) {
    direction = hipMemcpyHostToDevice;
    src = state->host_memory;
    dst = state->device_memory;
  } else {
    direction = hipMemcpyDeviceToHost;
    src = state->device_memory;
    dst = state->host_memory;
  }
  if (!CheckHIPError(hipMemcpyAsync(dst, src, action->data_size, direction,
    state->stream))) {
    return 0;
  }
  return 1;
}

static int Execute(void *data) {
  PluginState *state = (PluginState *) data;
  int i, kernel_number = 0;
  // Execute every action in order.
  for (i = 0; i < state->action_count; i++) {
    switch (state->actions[i].type) {
    case KERNEL_OPERATION:
      if (!LaunchKernelAction(state, i, kernel_number)) return 0;
      kernel_number++;
      break;
    case COPY_IN_OPERATION:
    case COPY_OUT_OPERATION:
      if (!LaunchCopyAction(state, i)) return 0;
      break;
    case SYNCHRONIZE_OPERATION:
      if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
      break;
    default:
      // Should be unreachable---caught when parsing the config.
      printf("Internal error: Invalid action type!\n");
      return 0;
    }
  }

  // Do a final stream synchronize.
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  PluginState *state = (PluginState *) data;
  KernelTimes *k = NULL;
  int i;
  // Copy out the block times for every kernel.
  for (i = 0; i < state->kernel_count; i++) {
    k = state->kernel_times + i;
    if (!CheckHIPError(hipMemcpyAsync(k->block_times,
      state->device_block_times[i], k->block_count * sizeof(uint64_t) * 2,
      hipMemcpyDeviceToHost, state->stream))) {
      return 0;
    }
  }
  times->kernel_times = state->kernel_times;
  times->kernel_count = state->kernel_count;
  // This resulting data is pretty meaningless, but may as well provide it.
  times->resulting_data_size = state->buffer_size;
  times->resulting_data = (void *) state->host_memory;

  // Make sure we're done copying block times.
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  return 1;
}

static const char* GetName(void) {
  return "Stream Actions";
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
