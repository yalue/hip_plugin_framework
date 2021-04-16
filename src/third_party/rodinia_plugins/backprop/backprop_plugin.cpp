// This has been modified into a HIP plugin in the following ways:
//  - It was ported from CUDA to HIP.
//  - The code has been reformatted.
//  - It required modifications to work with the plugin framework:
//    - Refactoring global variables
//    - Using HIP streams
//    - Move memory allocations and copies to functions apart from the
//      Execute function where possible.
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <hip/hip_runtime.h>
#include "plugin_hip_utilities.h"
#include "plugin_interface.h"
#include "plugin_utilities.h"

#include "backprop.h"
#include "backprop_hip_kernel.h"
#include "backprop_state.h"

// This is from the original Rodinia 3.1 benchmark's "run" file--it must be a
// multiple of 16.
#define LAYER_SIZE (65536)

void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);
void bpnn_output_error(float *delta, float *target, float *output, int nj,
    float *err);
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no,
    float **who, float *hidden, float *err);
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly,
    float **w, float **oldw);

float RandomFloat(PluginState *s) {
  double to_return;
  drand48_r(&(s->rng), &to_return);
  return to_return;
}

// The name "load" comes from the original benchmark--it appears to just
// randomly initialize the inputs.
static void Load(PluginState *s, BPNN *net) {
  float *units;
  int nr, i, k;

  nr = s->layer_size;
  units = net->input_units;
  k = 1;
  for (i = 0; i < nr; i++) {
    units[k] = RandomFloat(s);
    k++;
  }
}

// Like AllocateFloatArray, but allocates device memory.
static int AllocateFloatsDev(float **ptr, int n) {
  if (!CheckHIPError(hipMalloc(ptr, n * sizeof(float)))) return 0;
  return 1;
}

// Allocates and initializes memory used by the plugin.
static int AllocateMemory(PluginState *s) {
  BPNN *net = NULL;
  uint64_t *tmp = NULL;
  int in, hid, out, i;
  size_t block_times_size;
  net = BPNNCreate(s->layer_size, 16, 1);
  if (!net) return 0;
  s->net = net;
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;
  if (!AllocateFloatArray(&s->input_weights_one_dim, (in + 1) * (hid + 1))) {
    return 0;
  }
  if (!AllocateFloatArray(&s->input_weights_prev_one_dim, (in + 1) *
    (hid + 1))) {
    return 0;
  }
  s->num_blocks = in / 16;
  if (!AllocateFloatArray(&s->partial_sum, s->num_blocks * WIDTH)) {
    return 0;
  }
  if (!AllocateFloatsDev(&s->input_cuda, in + 1)) return 0;
  if (!AllocateFloatsDev(&s->output_hidden_cuda, hid + 1)) return 0;
  if (!AllocateFloatsDev(&s->hidden_delta_cuda, hid + 1)) return 0;
  if (!AllocateFloatsDev(&s->input_hidden_cuda, (in + 1) * (hid + 1))) {
    return 0;
  }
  if (!AllocateFloatsDev(&s->hidden_partial_sum, s->num_blocks * WIDTH)) {
    return 0;
  }
  if (!AllocateFloatsDev(&s->input_prev_weights_cuda, (in + 1) * (hid + 1))) {
    return 0;
  }
  // Both kernels use the same number of blocks--allocate space for the block
  // start and end times on the host and device.
  block_times_size = 2 * s->num_blocks * sizeof(uint64_t);
  for (i = 0; i < 2; i++) {
    if (!CheckHIPError(hipMalloc(&tmp, block_times_size))) return 0;
    s->device_block_times[i] = tmp;
    tmp = NULL;
    if (!CheckHIPError(hipHostMalloc(&tmp, block_times_size))) return 0;
    s->host_block_times[i] = tmp;
    tmp = NULL;
  }
  return 1;
}

static void Cleanup(void *data) {
  PluginState *s = (PluginState *) data;
  int i;
  if (s->net) BPNNFree(s->net);
  hipFree(s->input_cuda);
  hipFree(s->output_hidden_cuda);
  hipFree(s->hidden_delta_cuda);
  hipFree(s->input_hidden_cuda);
  hipFree(s->hidden_partial_sum);
  hipFree(s->input_prev_weights_cuda);
  hipHostFree(s->input_weights_one_dim);
  hipHostFree(s->input_weights_prev_one_dim);
  hipHostFree(s->partial_sum);
  for (i = 0; i < 2; i++) {
    hipFree(s->device_block_times[i]);
    hipHostFree(s->host_block_times[i]);
  }
  if (s->stream_created) {
    CheckHIPError(hipStreamDestroy(s->stream));
  }
  memset(s, 0, sizeof(*s));
  free(s);
}

static void* Initialize(InitializationParameters *params) {
  PluginState *s = NULL;
  int random_seed;
  if (!CheckHIPError(hipSetDevice(params->device_id))) return NULL;
  s = (PluginState *) calloc(1, sizeof(*s));
  if (!s) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }
  s->layer_size = LAYER_SIZE;
  if ((s->layer_size % 16) != 0) {
    printf("Layer size must be a multiple of 16.\n");
    free(s);
    return NULL;
  }

  // The original benchmark simply used 7 as a random seed, and used the non-
  // thread-safe srand() and rand() functions.
  random_seed = ((int) (CurrentSeconds() * 1e7) & 0x7fffffff);
  srand48_r(random_seed, &(s->rng));
  if (!AllocateMemory(s)) {
    Cleanup(s);
    return NULL;
  }
  // The thread count for these kernels is hardcoded to be dim3(16, 16). The
  // block count is dim3(1, num_blocks).
  s->kernel_times[0].kernel_name = "bpnn_layerforward_CUDA";
  s->kernel_times[0].thread_count = 16 * 16;
  s->kernel_times[0].block_count = s->num_blocks;
  s->kernel_times[0].block_times = s->host_block_times[0];
  s->kernel_times[1].kernel_name = "bpnn_adjust_weights_cuda";
  s->kernel_times[1].thread_count = 16 * 16;
  s->kernel_times[1].block_count = s->num_blocks;
  s->kernel_times[1].block_times = s->host_block_times[1];

  if (!CheckHIPError(CreateHIPStreamWithMask(&(s->stream),
    params->compute_unit_mask, COMPUTE_UNIT_MASK_ENTRIES))) {
    Cleanup(s);
    return NULL;
  }
  s->stream_created = 1;

  return s;
}

static int CopyIn(void *data) {
  int i, j, k, m, in, hid, out;
  PluginState *s = (PluginState *) data;
  BPNN *net = s->net;
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  // Re-initialize the data first.
  BPNNInitializeValues(s);
  Load(s, net);
  m = 0;
  for (k = 0; k <= in; k++) {
    for (j = 0; j <= hid; j++) {
      s->input_weights_one_dim[m] = net->input_weights[k][j];
      s->input_weights_prev_one_dim[m] = net->input_prev_weights[k][j];
      m++;
    }
  }

  // Now we can start copying.
  for (i = 0; i < 2; i++) {
    // Reset the block times to the maximum uint64_t values.
    if (!CheckHIPError(hipMemsetAsync(s->device_block_times[i], 0xff,
      2 * s->num_blocks * sizeof(uint64_t), s->stream))) {
      return 0;
    }
  }
  if (!CheckHIPError(hipMemcpyAsync(s->input_cuda, net->input_units, (in + 1) *
    sizeof(float), hipMemcpyHostToDevice, s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipMemcpyAsync(s->input_hidden_cuda,
    s->input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float),
    hipMemcpyHostToDevice, s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
  // Unfortunately, that's all the stuff we can copy before a kernel
  // invocation, so there are several copies in the Execute phase.
  return 1;
}

static int Execute(void *data) {
  int in, hid, out;
  PluginState *s = (PluginState *) data;
  BPNN *net = s->net;
  float out_err, hid_err;
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  // Boilerplate, etc, for launching the first kernel.
  s->kernel_times[0].kernel_launch_times[0] = CurrentSeconds();
  hipLaunchKernelGGL(bpnn_layerforward_CUDA, dim3(1, s->num_blocks),
    dim3(16, 16), 0, s->stream, s->input_cuda, s->output_hidden_cuda,
    s->input_hidden_cuda, s->hidden_partial_sum, in, hid,
    s->device_block_times[0]);
  s->kernel_times[0].kernel_launch_times[1] = CurrentSeconds();
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
  s->kernel_times[0].kernel_launch_times[2] = CurrentSeconds();

  // Copy out results of the first kernel.
  if (!CheckHIPError(hipMemcpyAsync(s->partial_sum, s->hidden_partial_sum,
    s->num_blocks * WIDTH * sizeof(float), hipMemcpyDeviceToHost,
    s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;

  // Now for some CPU computations before the second kernel.
  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights,
    hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out,
    &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
    net->hidden_weights, net->hidden_units, &hid_err);
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
    net->hidden_weights, net->hidden_prev_weights);

  // Now, we can copy data to the device for the second kernel.
  if (!CheckHIPError(hipMemcpyAsync(s->hidden_delta_cuda, net->hidden_delta,
    (hid + 1) * sizeof(float), hipMemcpyHostToDevice, s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipMemcpyAsync(s->input_prev_weights_cuda,
    s->input_weights_prev_one_dim, (in + 1) * (hid + 1) * sizeof(float),
    hipMemcpyHostToDevice, s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipMemcpyAsync(s->input_hidden_cuda,
    s->input_weights_one_dim, (in + 1) * (hid + 1) * sizeof(float),
    hipMemcpyHostToDevice, s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;

  // Finally, launch the second kernel.
  s->kernel_times[1].kernel_launch_times[0] = CurrentSeconds();
  hipLaunchKernelGGL(bpnn_adjust_weights_cuda, dim3(1, s->num_blocks),
    dim3(16, 16), 0, s->stream, s->hidden_delta_cuda, hid, s->input_cuda, in,
    s->input_hidden_cuda, s->input_prev_weights_cuda,
    s->device_block_times[1]);
  s->kernel_times[1].kernel_launch_times[1] = CurrentSeconds();
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
  s->kernel_times[1].kernel_launch_times[2] = CurrentSeconds();

  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  PluginState *s = (PluginState *) data;
  int in, hid, i;
  BPNN *net = s->net;
  in = net->input_n;
  hid = net->hidden_n;
  // Copy resulting data from the GPU.
  if (!CheckHIPError(hipMemcpyAsync(net->input_units, s->input_cuda, (in + 1) *
    sizeof(float), hipMemcpyDeviceToHost, s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipMemcpyAsync(s->input_weights_one_dim,
    s->input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float),
    hipMemcpyDeviceToHost, s->stream))) {
    return 0;
  }
  // Copy block times from the GPU (remember that kernel_times already has a
  // copy of the pointer to this data).
  for (i = 0; i < 2; i++) {
    if (!CheckHIPError(hipMemcpyAsync(s->host_block_times[i],
      s->device_block_times[i], 2 * s->num_blocks * sizeof(uint64_t),
      hipMemcpyDeviceToHost, s->stream))) {
      return 0;
    }
  }
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
  // Provide the timing data to the caller.
  times->kernel_count = 2;
  times->kernel_times = s->kernel_times;
  times->resulting_data_size = 0;
  times->resulting_data = NULL;
  return 1;
}

static const char* GetName(void) {
  return "Backprop (Rodinia)";
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

