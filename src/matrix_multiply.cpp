// This plugin multiplies two square matrices, using a not-particularly-
// optimized matrix multiply kernel. This uses a 2D grid of 2D blocks, mapping
// one thread per resulting matrix element. The block_count setting from the
// config is ignored, and thread_count must be either 64, 256, or 1024, in
// order to use 8x8, 16x16, or 32x32 blocks, respectively. The additional_info
// must be a JSON object with the following keys:
//
//  - "matrix_width": The width of the square matrix of floating-point numbers
//    to multiply.
//
// - "skip_copy": A boolean. Optional, defaults to false. If it's set to true,
//   then the "copy_in" and "copy_out" phases will not copy the matrix data to
//   or from the GPU. Instead, the input matrices will only be copied once,
//   during the initialize function, and output data will never be copied. May
//   be useful if you want a simple workload without as many memory transfers.

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
  // The grid size is computed based on the matrix width.
  dim3 grid_size;
  // The block size is determined by the thread_count setting in the config.
  dim3 block_size;
  // Set this to nonzero in order to not copy input or output matrices, apart
  // from during initialization.
  int skip_copy;
  // The width of each square matrix.
  int matrix_width;
  // The device-side matrices. The computation will be d_c = d_a x d_b.
  float *d_a, *d_b, *d_c;
  // The three host-side copies of the matrices.
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

static float RandomFloat(void) {
  // Maybe replace this with something faster?
  float to_return = ((float) rand()) / ((float) RAND_MAX);
  return to_return;
}

// Allocates the host and device matrices, and randomly initializes them.
// Returns 0 on error. Must be called after grid_size has been initialized.
static int AllocateMemory(PluginState *state) {
  int i, j, width, block_count;
  size_t size;
  width = state->matrix_width;
  block_count = state->grid_size.x * state->grid_size.y;

  // Allocate host and device memory for block times.
  size = block_count * 2 * sizeof(uint64_t);
  if (!CheckHIPError(hipHostMalloc(&(state->kernel_times.block_times),
    size))) {
    return 0;
  }
  if (!CheckHIPError(hipMalloc(&(state->device_block_times), size))) {
    return 0;
  }

  // Allocate the matrices.
  size = width * width * sizeof(float);
  if (!CheckHIPError(hipMalloc(&state->d_a, size))) return 0;
  if (!CheckHIPError(hipMalloc(&state->d_b, size))) return 0;
  if (!CheckHIPError(hipMalloc(&state->d_c, size))) return 0;
  if (!CheckHIPError(hipHostMalloc(&state->h_a, size))) return 0;
  if (!CheckHIPError(hipHostMalloc(&state->h_b, size))) return 0;
  if (!CheckHIPError(hipHostMalloc(&state->h_c, size))) return 0;
  memset(state->h_c, 0, size);

  // Randomly initialize the host's input matrices.
  for (i = 0; i < width; i++) {
    for (j = 0; j < width; j++) {
      state->h_a[i * width + j] = RandomFloat();
      state->h_b[i * width + j] = RandomFloat();
    }
  }

  // Initialize the input matrices on the host.
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
    printf("Invalid additional_info for matrix_multiply.\n");
    return 0;
  }

  // Make sure that matrix_width is present and positive.
  entry = cJSON_GetObjectItem(root, "matrix_width");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Invalid matrix_width setting.\n");
    cJSON_Delete(root);
    return 0;
  }
  if (entry->valuedouble < 1.0) {
    printf("The matrix_width setting must be at least 1.\n");
    cJSON_Delete(root);
    return 0;
  }
  if (entry->valuedouble > 1e6) {
    printf("Warning: huge matrix width: %f\n", entry->valuedouble);
  }
  state->matrix_width = (int) entry->valuedouble;

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
  int blocks_wide, matrix_width, thread_count;
  if (!CheckHIPError(hipSetDevice(params->device_id))) return NULL;
  if (!CheckHIPError(hipSetDeviceFlags(PLUGIN_DEVICE_FLAGS))) return NULL;
  state = (PluginState *) calloc(sizeof(*state), 1);
  if (!state) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }

  // We need to know the matrix width before we can allocate memory or
  // determine grid size.
  if (!ParseAdditionalInfo(params->additional_info, state)) {
    Cleanup(state);
    return NULL;
  }
  matrix_width = state->matrix_width;

  // Ensure that the thread_count allows us to form a square thread block.
  thread_count = params->thread_count;
  switch (thread_count) {
  case 64:
    state->block_size.x = 8;
    state->block_size.y = 8;
    break;
  case 256:
    state->block_size.x = 16;
    state->block_size.y = 16;
    break;
  case 1024:
    state->block_size.x = 32;
    state->block_size.y = 32;
    break;
  default:
    printf("Unsupported matrix_multiply thread_count: %d\n", thread_count);
    Cleanup(state);
    return NULL;
  }
  state->block_size.z = 1;

  // Compute the grid size from the block size and matrix width.
  blocks_wide = matrix_width / state->block_size.x;
  if ((matrix_width % state->block_size.x) != 0) blocks_wide++;
  state->grid_size.x = blocks_wide;
  state->grid_size.y = blocks_wide;
  state->grid_size.z = 1;

  // Create the stream and fill in boilerplate for reporting to the framework.
  if (!CheckHIPError(CreateHIPStreamWithMask(&(state->stream),
    params->compute_unit_mask, COMPUTE_UNIT_MASK_ENTRIES))) {
    Cleanup(state);
    return NULL;
  }
  state->stream_created = 1;
  state->kernel_times.kernel_name = "matrix_multiply";
  state->kernel_times.thread_count = state->block_size.x * state->block_size.y;
  state->kernel_times.block_count = state->grid_size.x * state->grid_size.y;
  state->kernel_times.shared_memory = 0;

  // Allocate the matrices and initialize the input matrices
  if (!AllocateMemory(state)) {
    Cleanup(state);
    return NULL;
  }
  return state;
}

// Reset block times and copy input matrices.
static int CopyIn(void *data) {
  PluginState *state = (PluginState *) data;
  int block_count = state->grid_size.x * state->grid_size.y;

  // Reset block times
  size_t size = block_count * 2 * sizeof(uint64_t);
  if (!CheckHIPError(hipMemsetAsync(state->device_block_times, 0xff,
    size, state->stream))) {
    return 0;
  }

  // If we're skipping the copies we can return now.
  if (state->skip_copy) {
    if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
    return 1;
  }

  // Copy input matrices.
  size = state->matrix_width * state->matrix_width * sizeof(float);
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

static int Execute(void *data) {
  PluginState *state = (PluginState *) data;
  state->kernel_times.kernel_launch_times[0] = CurrentSeconds();

  hipLaunchKernelGGL(MatrixMultiplyKernel, state->grid_size, state->block_size,
    0, state->stream, state->d_a, state->d_b, state->d_c, state->matrix_width,
    state->device_block_times);

  state->kernel_times.kernel_launch_times[1] = CurrentSeconds();
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  state->kernel_times.kernel_launch_times[2] = CurrentSeconds();
  return 1;
}

// Copy the block times to the host, along with the result matrix.
static int CopyOut(void *data, TimingInformation *times) {
  PluginState *state = (PluginState *) data;
  int block_count = state->grid_size.x * state->grid_size.y;
  size_t size;


  // First copy the block times to the host.
  size = block_count * 2 * sizeof(uint64_t);
  if (!CheckHIPError(hipMemcpyAsync(state->kernel_times.block_times,
    state->device_block_times, size, hipMemcpyDeviceToHost, state->stream))) {
    return 0;
  }

  // Provide the framework with a pointer to our kernel_times struct.
  times->kernel_count = 1;
  times->kernel_times = &(state->kernel_times);

  // We can return now if we're not copying the result matrix.
  if (state->skip_copy) {
    times->resulting_data_size = 0;
    times->resulting_data = NULL;
    if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
    return 1;
  }

  // Copy the result matrix.
  size = state->matrix_width * state->matrix_width * sizeof(float);
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
  return "Matrix Multiply";
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

