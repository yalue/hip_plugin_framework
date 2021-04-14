// This plugin renders a black-and-white mandelbrot set image on the GPU. Its
// "additional_info" parameter is optional, but if present must contain a JSON
// object. The JSON object may contain two keys: "max_iterations" and
// "image_width", both of which are optional and fall back to sane defaults if
// unspecified.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include "third_party/cJSON.h"
#include "plugin_hip_utilities.h"
#include "plugin_interface.h"
#include "plugin_utilities.h"

#define DEFAULT_MAX_ITERATIONS (1000)

// The resulting image will be a square this number of pixels wide. Can be overriden
#define DEFAULT_PIXELS_WIDE (512)

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
  // the mandelbrot set) or 1 (escapes the set).
  uint8_t *host_image;
  uint8_t *device_image;

  // Keep track of the boundaries of the complex plane and image dimensions, as
  // well as the max iterations when drawing the mandelbrot set.
  uint64_t max_iterations;
  FractalDimensions dimensions;

  // The grid and block dimensions. The block count is determined by the
  // thread_count initialization parameter and the image width (we use one
  // thread per pixel).
  int block_count;
  int thread_count;

  // A buffer to hold the block times on the device (the host-side copy of this
  // buffer will be referenced in the KernelTimes struct).
  uint64_t *device_block_times;

  // Keeps track of per-kernel timing data required by the plugin framework.
  // This plugin only has one kernel per iteration, so we only need one of
  // these structs.
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

// Parses the additional_info argument to optionally override defaults. Returns
// 0 on error.
static int ParseAdditionalInfo(const char *arg, PluginState *state) {
  cJSON *root = NULL;
  cJSON *entry = NULL;
  // It's fine for the additional_info field to not be present--just leave the
  // defaults.
  if (!arg || (strlen(arg) == 0)) return 1;

  // The additional_info field must be a JSON object if present, so try to
  // parse it as JSON.
  root = cJSON_Parse(arg);
  if (!root) {
    printf("Invalid additional_info for mandelbrot.\n");
    return 0;
  }
  if (root->type != cJSON_Object) {
    printf("additional_info for mandelbrot must be a JSON object.\n");
    cJSON_Delete(root);
    return 0;
  }

  // Read the max_iterations setting. If it's there, then it must be a number.
  // If it's not there, we'll just leave state->max_iterations at its current
  // (default) value.
  entry = cJSON_GetObjectItem(root, "max_iterations");
  if (entry) {
    if (entry->type != cJSON_Number) {
      printf("max_iterations setting for mandelbrot must be a number.\n");
      cJSON_Delete(root);
      return 0;
    }
    // Using more than 2 billion iterations would make little sense, anyway.
    if (entry->valueint <= 0) {
      printf("Invalid max iterations for mandelbrot: %d\n", entry->valueint);
      cJSON_Delete(root);
      return 0;
    }
    state->max_iterations = entry->valueint;
  }

  // Do the same stuff for the image_width setting as for max_iterations.
  entry = cJSON_GetObjectItem(root, "image_width");
  if (entry) {
    if (entry->type != cJSON_Number) {
      printf("image_width setting for mandelbrot must be a number.\n");
      cJSON_Delete(root);
      return 0;
    }
    if (entry->valueint <= 0) {
      printf("Invalid image width for mandelbrot: %d\n", entry->valueint);
      cJSON_Delete(root);
      return 0;
    }
    state->dimensions.w = entry->valueint;
    state->dimensions.h = entry->valueint;
  }

  // We don't need the parsed JSON now that we've read the settings.
  cJSON_Delete(root);
  root = NULL;
  entry = NULL;
  return 1;
}

// Allocates host and device memory for the plugin. Returns 0 on error.
static int AllocateMemory(PluginState *state) {
  // First, allocate a buffer to hold the mandelbrot "image" on the host and
  // device. It's a binary image with one byte per pixel, for simplicity.
  size_t buffer_size = state->dimensions.w * state->dimensions.h;
  if (!CheckHIPError(hipHostMalloc(&(state->host_image), buffer_size))) {
    return 0;
  }
  if (!CheckHIPError(hipMalloc(&(state->device_image), buffer_size))) return 0;

  // Next, allocate a buffer to hold block start and end times. Remember that
  // the pointer to host-side times is in the KernelTimes struct...
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
  if (!CheckHIPError(hipSetDeviceFlags(PLUGIN_DEVICE_FLAGS))) return NULL;

  // Allocate memory to hold state for this instance of the plugin.
  state = (PluginState *) calloc(1, sizeof(*state));
  if (!state) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }

  // Set the default values, and those specified in the overall config.
  if (!GetSingleBlockDimension(params, &state->thread_count)) {
    Cleanup(state);
    return NULL;
  }
  dimensions = &(state->dimensions);
  dimensions->w = DEFAULT_PIXELS_WIDE;
  dimensions->h = DEFAULT_PIXELS_WIDE;
  dimensions->min_real = -2.0;
  dimensions->max_real = 2.0;
  dimensions->min_imag = -2.0;
  dimensions->max_imag = 2.0;
  dimensions->delta_real = 4.0 / dimensions->w;
  dimensions->delta_imag = 4.0 / dimensions->h;
  state->max_iterations = DEFAULT_MAX_ITERATIONS;
  if ((state->thread_count <= 0) || (state->thread_count > 1024)) {
    printf("Invalid thread count.\n");
    Cleanup(state);
    return NULL;
  }

  // Deal with any additional_info if it was provided.
  if (!ParseAdditionalInfo(params->additional_info, state)) {
    Cleanup(state);
    return NULL;
  }

  // Compute the block count, adding an additional block if the number of
  // pixels isn't evenly divisible by thread_count.
  state->block_count = (dimensions->w * dimensions->h) / state->thread_count;
  if (((dimensions->w * dimensions->h) % state->thread_count) != 0) {
    state->block_count++;
  }

  // Wrap up by creating the stream and setting the kernel_times fields that
  // don't change.
  if (!CheckHIPError(CreateHIPStreamWithMask(&(state->stream),
    params->compute_unit_mask, COMPUTE_UNIT_MASK_ENTRIES))) {
    Cleanup(state);
    return NULL;
  }
  state->stream_created = 1;
  state->kernel_times.kernel_name = "mandelbrot";
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

  // We're setting every uint64_t in the device_block_times array to its max
  // value.
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

  // This requires always initializing block times to the maximum integer (done
  // in CopyIn), but it  should ensure we have something very close to the
  // earliest possible start time for a given block.
  if (start_clock < block_times[hipBlockIdx_x * 2]) {
    block_times[hipBlockIdx_x * 2] = start_clock;
  }

  // Figure out offset of the byte corresponding to the pixel we're computing,
  // and return early if it's outside the image (as may be the case if the
  // number of pixels isn't evenly divisible by the number of threads per
  // block).
  buffer_size = dims.w * dims.h;
  index = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
  if (index > buffer_size) {
    block_times[hipBlockIdx_x * 2 + 1] = clock64();
    return;
  }

  // Figure out the coordinate of the pixel on the complex plane.
  row = index / dims.w;
  col = index % dims.h;
  start_real = dims.min_real + dims.delta_real * col;
  start_imag = dims.min_imag + dims.delta_imag * row;
  real = start_real;
  imag = start_imag;
  // "magnitude" is actually the magnitude squared, to avoid the need for a
  // square root operation.
  magnitude = (real * real) + (imag * imag);
  // Actually run the mandelbrot-set iterations
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

  // Set the pixel, which will be 1 if the value escaped.
  buffer[index] = escaped;

  // This will always be set by the last thread to execute, or at least by a
  // thread that is very close to the last one.
  block_times[hipBlockIdx_x * 2 + 1] = clock64();
}

static int Execute(void *data) {
  PluginState *state = (PluginState *) data;
  // In addition to block times, we record 3 times per kernel launch: the time
  // immediately before the async launch, the time immediately after the async
  // launch, and the time after hipStreamSynchronize returns, indicating the
  // kernel has completed.
  state->kernel_times.kernel_launch_times[0] = CurrentSeconds();

  // Here is where we actually launch the kernel.
  hipLaunchKernelGGL(MandelbrotKernel, state->block_count, state->thread_count,
    0, state->stream, state->device_image, state->dimensions,
    state->max_iterations, state->device_block_times);
  state->kernel_times.kernel_launch_times[1] = CurrentSeconds();

  // Wait for the kernel to complete.
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  state->kernel_times.kernel_launch_times[2] = CurrentSeconds();
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  PluginState *state = (PluginState *) data;
  size_t image_size = state->dimensions.w * state->dimensions.h;

  // The caller generally just ignores this data, but if you want to dump it to
  // a ".pbm" image starting with a "P4" identifier, it's pretty easy--check
  // wikipedia's "Netpbm" article. The host_image buffer can be dumped as the
  // image data.
  if (!CheckHIPError(hipMemcpyAsync(state->host_image, state->device_image,
    image_size, hipMemcpyDeviceToHost, state->stream))) {
    return 0;
  }

  // We need to copy the block times to where the host can read them. The
  // interface states that the contents of the host's block_times buffer only
  // needs to remain valid until another plugin function is called, so it's
  // fine to overwrite any old data in the buffer by the time we get to this
  // point.
  if (!CheckHIPError(hipMemcpyAsync(state->kernel_times.block_times,
    state->device_block_times, state->block_count * 2 * sizeof(uint64_t),
    hipMemcpyDeviceToHost, state->stream))) {
    return 0;
  }

  // The TimingInformation struct requires a pointer to an array of KernelTimes
  // structs. Our "array" only consists of one such struct. This plugin also
  // sets the optional "resulting_data" field.
  times->kernel_count = 1;
  times->kernel_times = &(state->kernel_times);
  times->resulting_data_size = image_size;
  times->resulting_data = state->host_image;

  // Make sure we're done copying everything before returning. (Notice how all
  // plugins should be careful to keep their associated copies and kernel
  // launches limited to instance-specific streams.)
  if (!CheckHIPError(hipStreamSynchronize(state->stream))) return 0;
  return 1;
}

// Trivial boilerplate.
static const char* GetName(void) {
  return "Mandelbrot";
}

int RegisterPlugin(PluginFunctions *functions) {
  // Keeping only one exported function with a struct of function pointers
  // means a lot fewer calls to "dlsym".
  functions->get_name = GetName;
  functions->cleanup = Cleanup;
  functions->initialize = Initialize;
  functions->copy_in = CopyIn;
  functions->execute = Execute;
  functions->copy_out = CopyOut;
  return 1;
}
