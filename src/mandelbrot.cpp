// This plugin renders a black-and-white mandelbrot set image on the GPU. It
// takes a single "additional_info" parameter specifying the maximum number of
// iterations to use, but this parameter is optional and defaults to 1000. The
// size of the image to create is determined by the square root of the
// data_size option (which roughly specifies the size in bytes of a square
// image).
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include "plugin_hip_utilities.h"
#include "plugin_interface.h"
#include "plugin_utilities.h"

#define DEFAULT_MAX_ITERATIONS (1000)

// The resulting mandelbrot set will be a square image of this size. (This may
// be modifiable using an additional_info parameter later.)
#define IMAGE_PIXELS_WIDE (256)

// Holds the boundaries and size of the fractal, in both pixels and in terms
// of the complex bounding box.
typedef struct {
  // The width and height of the image in pixels.
  int w;
  int h;
  // The boundaries of the fractal on the complex plane.
  double min_real;
  double max_real;
  double min_imag;
  double max_imag;
  // The distance between pixels in the real and imaginary axes.
  double delta_real;
  double delta_imag;
} FractalDimensions;

// Holds the state of an instance of this plugin.
typedef struct {
  // The stream that this plugin will use
  hipStream_t stream;
  // This will be 0 if the stream hasn't been created yet. This helps keep
  // cleanup code simpler.
  int stream_created;
  // Holds the bitmaps on the host and device. Each value will be either 0 (in
  // the set) or 1 (escapes the set).
  uint8_t *host_image;
  uint8_t *device_image;
  uint64_t max_iterations;
  FractalDimensions dimensions;
  // The grid and block dimensions. The block count is determined by the
  // data_size and thread_count initialization parameters.
  int block_count;
  int thread_count;
  uint64_t *device_block_times;
  // Holds times that are shared with the plugin host.
  KernelTimes kernel_times;
} PluginState;

// Implements the Cleanup() function required by the plugin interface.
static void Cleanup(void *data) {
  PluginState *state = (PluginState *) data;
  if (state->device_image) hipFree(state->device_image);
  if (state->host_image) hipHostFree(state->host_image);
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

// Parses the additional_info argument to see if it contains a number to
// override the max iterations. Returns 0 on error.
static int SetMaxIterations(const char *arg, PluginState *state) {
  int64_t parsed_value;
  char *end = NULL;
  if (!arg || (strlen(arg) == 0)) {
    state->max_iterations = DEFAULT_MAX_ITERATIONS;
    return 1;
  }
  parsed_value = strtoll(arg, &end, 10);
  if ((*end != 0) || (parsed_value < 0)) {
    printf("Invalid max iterations: %s\n", arg);
    return 0;
  }
  state->max_iterations = (uint64_t) parsed_value;
  return 1;
}

// Allocates host and device memory for the plugin. Returns 0 on error.
static int AllocateMemory(PluginState *state) {
  size_t buffer_size = state->dimensions.w * state->dimensions.h;
  if (!CheckHIPError(hipHostMalloc(&(state->host_image), buffer_size))) {
    return 0;
  }
  if (!CheckHIPError(hipMalloc(&(state->device_image), buffer_size))) return 0;
  // Next, allocate a buffer to hold block start and end times.
  buffer_size = state->block_count * 2 * sizeof(uint64_t);
  if (!CheckHIPError(hipMalloc(&(state->device_block_times), buffer_size))) {
    return 0;
  }
  if (!CheckHIPError(hipHostMalloc(&(state->kernel_times.block_times),
    buffer_size))) {
    return 0;
  }
  return 1;
}

static void* Initialize(InitializationParameters *params) {
  PluginState *state = NULL;
  FractalDimensions *dimensions = NULL;
  if (!CheckHIPError(hipSetDevice(params->device_id))) return NULL;
  state = (PluginState *) malloc(sizeof(*state));
  if (!state) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }
  memset(state, 0, sizeof(*state));
  state->thread_count = params->thread_count;
  dimensions = &(state->dimensions);
  dimensions->w = IMAGE_PIXELS_WIDE;
  dimensions->h = IMAGE_PIXELS_WIDE;
  if ((dimensions->w == 0) || (dimensions->h == 0)) {
    // TODO: This will be important only later if the width and height can be
    // configured by the user.
    printf("Bad Mandelbrot image data size.\n");
    Cleanup(state);
    return NULL;
  }
  dimensions->min_real = -2.0;
  dimensions->max_real = 2.0;
  dimensions->min_imag = -2.0;
  dimensions->max_imag = 2.0;
  dimensions->delta_real = 4.0 / dimensions->w;
  dimensions->delta_imag = 4.0 / dimensions->h;
  if ((params->thread_count <= 0) || (params->thread_count > 1024)) {
    printf("Invalid thread count.\n");
    Cleanup(state);
    return NULL;
  }
  state->block_count = (dimensions->w * dimensions->h) / params->thread_count;
  // Add one more block when the image isn't evenly divisible by thread_count.
  if (((dimensions->w * dimensions->h) % params->thread_count) != 0) {
    state->block_count++;
  }
  if (!SetMaxIterations(params->additional_info, state)) {
    Cleanup(state);
    return NULL;
  }
  if (!CheckHIPError(CreateHIPStreamWithMask(&(state->stream),
    params->compute_unit_mask, COMPUTE_UNIT_MASK_ENTRIES))) {
    Cleanup(state);
    return NULL;
  }
  state->stream_created = 1;
  state->kernel_times.kernel_name = "mandelbrot";
  state->kernel_times.thread_count = params->thread_count;
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

// The kernel for rendering the mandelbrot set. Not very optimized, but that's
// not important for a simple benchmarking use case.
__global__ void MandelbrotKernel(uint8_t *buffer, FractalDimensions dims,
  uint64_t max_iterations, uint64_t *block_times) {
  uint64_t i;
  double start_real, start_imag, real, imag, tmp, magnitude;
  int buffer_size, index, row, col;
  uint8_t escaped = 0;
  uint64_t start_clock = clock64();
  // This requires always initializing block times to the maximum integer, but
  // should ensure we have something very close to the earliest possible start
  // time for a given block.
  if (start_clock < block_times[hipBlockIdx_x * 2]) {
    block_times[hipBlockIdx_x * 2] = start_clock;
  }
  buffer_size = dims.w * dims.h;
  index = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
  if (index > buffer_size) return;
  row = index / dims.w;
  col = index % dims.h;
  start_real = dims.min_real + dims.delta_real * col;
  start_imag = dims.min_imag + dims.delta_imag * row;
  real = start_real;
  imag = start_imag;
  magnitude = (real * real) + (imag * imag);
  // "magnitude" is actually the magnitude squared, to avoid the need for a
  // square root operation.
  for (i = 0; i < max_iterations; i++) {
    if (magnitude > 4) {
      escaped = 1;
      break;
    }
    tmp = (real * real) - (imag * imag);
    imag = 2 * real * imag + start_imag;
    real = tmp;
    magnitude = (real * real) + (imag * imag);
  }
  buffer[index] = escaped;
  // This will always be set by the last thread to execute, or at least by a
  // thread that is very close to the last one.
  block_times[hipBlockIdx_x * 2 + 1] = clock64();
}

static int Execute(void *data) {
  PluginState *state = (PluginState *) data;
  state->kernel_times.kernel_launch_times[0] = CurrentSeconds();
  hipLaunchKernelGGL(MandelbrotKernel, state->block_count, state->thread_count,
    0, state->stream, state->device_image, state->dimensions,
    state->max_iterations, state->device_block_times);
  state->kernel_times.kernel_launch_times[1] = CurrentSeconds();
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  state->kernel_times.kernel_launch_times[2] = CurrentSeconds();
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  PluginState *state = (PluginState *) data;
  size_t image_size = state->dimensions.w * state->dimensions.h;
  if (!CheckHIPError(hipMemcpyAsync(state->host_image, state->device_image,
    image_size, hipMemcpyDeviceToHost, state->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipMemcpyAsync(state->kernel_times.block_times,
    state->device_block_times, state->block_count * 2 * sizeof(uint64_t),
    hipMemcpyDeviceToHost, state->stream))) {
    return 0;
  }
  times->kernel_count = 1;
  times->kernel_times = &(state->kernel_times);
  times->resulting_data_size = image_size;
  times->resulting_data = state->host_image;
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  return 1;
}

static const char* GetName(void) {
  return "Mandelbrot";
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
