/*-----------------------------------------------------------
 ** gaussian.cu -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.  
 **   The sequential version is gaussian.c.  This parallel 
 **   implementation converts three independent for() loops 
 **   into three Fans.  Use the data file ge_3.dat to verify 
 **   the correction of the output. 
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 **-----------------------------------------------------------
 */

// This has been modified into a HIP plugin in the following ways:
//  - It was ported from CUDA to HIP.
//  - The code has been reformatted.
//  - It required several modifications to work with the plugin framework:
//    - Refactoring to not use global variables
//    - Kernels and memory transfers use HIP streams
//    - Memory allocations were all moved into the Initialize function

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include "plugin_hip_utilities.h"
#include "plugin_interface.h"
#include "plugin_utilities.h"

// These are just default values of defines from the original file.
#define MAXBLOCKSIZE (512)
#define BLOCK_SIZE_XY (4)

// This was supplied as a command-line "-s" argument in the original, but we'll
// hardcode it here to the value that was in Rodinia 3.1's "run" file
// associated with the benchmark.
#define MATRIX_SIZE (16)

typedef struct {
  // Fields used in basically all plugins, see one of the others for details.
  hipStream_t stream;
  int stream_created;
  KernelTimes *kernel_times;
  // Stuff that used to be globals in the original gaussian benchmark.
  int Size;
  float *m, *a, *b, *finalVec;
  // Avoid reallocating and freeing device memory by tracking it here, another
  // change from the original benchmark.
  float *m_device, *a_device, *b_device;
} PluginState;

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
static void CreateMatrix(float *m, int size) {
  int i, j;
  float lamda = -0.01;
  float coe[2 * size - 1];
  float coe_i = 0.0;
  for (i = 0; i < size; i++) {
    coe_i = 10 * exp(lamda * i);
    j = size - 1 + i;
    coe[j] = coe_i;
    j = size - 1 - i;
    coe[j] = coe_i;
  }
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++) {
      m[i * size + j] = coe[size - 1 - i + j];
    }
  }
}

// Allocates host and device memory. Returns 0 on error.
static int AllocateMemory(PluginState *s) {
  size_t size = s->Size * s->Size * sizeof(float);

  // We'll leave initialization of most of this memory until CopyIn, since it
  // will need to be done before every iteration.
  if (!CheckHIPError(hipHostMalloc(&(s->a), size))) {
    return 0;
  }
  if (!CheckHIPError(hipHostMalloc(&(s->b), s->Size * sizeof(float)))) {
    return 0;
  }
  if (!CheckHIPError(hipHostMalloc(&(s->finalVec), s->Size * sizeof(float)))) {
    return 0;
  }
  if (!CheckHIPError(hipHostMalloc(&(s->m), size))) {
    return 0;
  }
  // This benchmark invokes 2 different kernels per iteration in (s->Size - 1)
  // iterations.
  if (!CheckHIPError(hipHostMalloc(&(s->kernel_times), 2 * (s->Size - 1) *
    sizeof(KernelTimes)))) {
    return 0;
  }
  memset(s->kernel_times, 0, 2 * (s->Size - 1) * sizeof(KernelTimes));

  if (!CheckHIPError(hipMalloc(&(s->a_device), size))) {
    return 0;
  }
  if (!CheckHIPError(hipMalloc(&(s->b_device), s->Size * sizeof(float)))) {
    return 0;
  }
  if (!CheckHIPError(hipMalloc(&(s->m_device), size))) {
    return 0;
  }

  return 1;
}

static void Cleanup(void *data) {
  PluginState *s = (PluginState *) data;
  if (s->m) hipHostFree(s->m);
  if (s->a) hipHostFree(s->a);
  if (s->b) hipHostFree(s->b);
  if (s->finalVec) hipHostFree(s->finalVec);
  if (s->kernel_times) hipHostFree(s->kernel_times);
  if (s->m_device) hipFree(s->m_device);
  if (s->a_device) hipFree(s->a_device);
  if (s->b_device) hipFree(s->b_device);
  if (s->stream_created) {
    CheckHIPError(hipStreamDestroy(s->stream));
  }
  memset(s, 0, sizeof(*s));
  free(s);
}

static void* Initialize(InitializationParameters *params) {
  PluginState *s = NULL;
  int block_count, block_count_2d, i;
  if (!CheckHIPError(hipSetDevice(params->device_id))) {
    return NULL;
  }
  s = (PluginState *) malloc(sizeof(*s));
  if (!s) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }
  memset(s, 0, sizeof(*s));
  s->Size = MATRIX_SIZE;
  if (!AllocateMemory(s)) {
    Cleanup(s);
    return NULL;
  }
  if (!CheckHIPError(CreateHIPStreamWithMask(&(s->stream),
    params->compute_unit_mask, COMPUTE_UNIT_MASK_ENTRIES))) {
    Cleanup(s);
    return NULL;
  }
  s->stream_created = 1;
  block_count = s->Size / MAXBLOCKSIZE;
  if ((s->Size % MAXBLOCKSIZE) != 0) block_count++;
  block_count_2d = s->Size / BLOCK_SIZE_XY;
  if ((s->Size % BLOCK_SIZE_XY) != 0) block_count_2d++;
  // Fan2's uses a 2D grid of blocks and threads.
  block_count_2d *= block_count_2d;
  // Pre-fill in the information about each kernel, there's no need to copy all
  // the same stuff every iteration.
  for (i = 0; i < (s->Size - 1); i++) {
    s->kernel_times[i * 2].kernel_name = "Fan1";
    s->kernel_times[i * 2].thread_count = MAXBLOCKSIZE;
    s->kernel_times[i * 2].block_count = block_count;
    s->kernel_times[i * 2 + 1].kernel_name = "Fan2";
    s->kernel_times[i * 2 + 1].thread_count = BLOCK_SIZE_XY * BLOCK_SIZE_XY;
    s->kernel_times[i * 2 + 1].block_count = block_count_2d;
  }
  return s;
}

static int CopyIn(void *data) {
  PluginState *s = (PluginState *) data;
  size_t size = s->Size * s->Size * sizeof(float);
  int j;

  // First, we'll re-initialize memory here so that Execute does the same thing
  // every time.
  CreateMatrix(s->a, s->Size);
  for (j = 0; j < s->Size; j++) s->b[j] = 1.0;
  memset(s->finalVec, 0, s->Size * sizeof(float));
  memset(s->m, 0, size);

  // Now copy the clean data to the device.
  if (!CheckHIPError(hipMemcpyAsync(s->m_device, s->m, size,
    hipMemcpyHostToDevice, s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipMemcpyAsync(s->a_device, s->a, size,
    hipMemcpyHostToDevice, s->stream))) {
    return 0;
  }
  size = s->Size * sizeof(float);
  if (!CheckHIPError(hipMemcpyAsync(s->b_device, s->b, size,
    hipMemcpyHostToDevice, s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
  return 1;
}

/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */
__global__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t) {
  if ((blockIdx.x * blockDim.x + threadIdx.x) >= (Size - 1 - t)) return;
  *(m_cuda + Size * (blockDim.x * blockIdx.x + threadIdx.x + t + 1) + t) =
    *(a_cuda + Size * (blockDim.x * blockIdx.x + threadIdx.x + t + 1) + t) /
    *(a_cuda + Size * t + t);
}

/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */
__global__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda,int Size,
  int j1, int t) {
  if ((blockIdx.x * blockDim.x + threadIdx.x) >= (Size - 1 - t)) return;
  if ((blockIdx.y * blockDim.y + threadIdx.y) >= (Size - t)) return;
  int xidx = blockIdx.x * blockDim.x + threadIdx.x;
  int yidx = blockIdx.y * blockDim.y + threadIdx.y;

  a_cuda[Size * (xidx + 1 + t) + (yidx + t)] -=
    m_cuda[Size * (xidx + 1 + t) + t] * a_cuda[Size * t + (yidx + t)];
  if (yidx == 0) {
    b_cuda[xidx + 1 + t] -= m_cuda[Size * (xidx + 1 + t) + (yidx + t)] *
      b_cuda[t];
  }
}

static int Execute(void *data) {
  PluginState *s = (PluginState *) data;
  int block_size, grid_size, block_size_2d, grid_size_2d, t;
  block_size = MAXBLOCKSIZE;
  grid_size = (s->Size / block_size) + (!(s->Size % block_size) ? 0 : 1);
  block_size_2d = BLOCK_SIZE_XY;
  grid_size_2d = (s->Size / block_size_2d) + (!(s->Size % block_size_2d) ?
    0 : 1);
  for (t = 0; t < (s->Size - 1); t++) {
    // Run the first kernel, surrounded by a bunch of timing bookkeeping.
    s->kernel_times[t * 2].kernel_launch_times[0] = CurrentSeconds();
    hipLaunchKernelGGL(Fan1, dim3(grid_size), dim3(block_size), 0, s->stream,
      s->m_device, s->a_device, s->Size, t);
    s->kernel_times[t * 2].kernel_launch_times[1] = CurrentSeconds();
    if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
    s->kernel_times[t * 2].kernel_launch_times[2] = CurrentSeconds();
    // Now run the second kernel.
    s->kernel_times[t * 2 + 1].kernel_launch_times[0] = CurrentSeconds();
    hipLaunchKernelGGL(Fan2, dim3(grid_size_2d, grid_size_2d),
      dim3(block_size_2d, block_size_2d), 0, s->stream, s->m_device,
      s->a_device, s->b_device, s->Size, s->Size - t, t);
    s->kernel_times[t * 2 + 1].kernel_launch_times[1] = CurrentSeconds();
    if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
    s->kernel_times[t * 2 + 1].kernel_launch_times[2] = CurrentSeconds();
  }
  return 1;
}

// This fills in finalVec and requires that the data has already been copied
// back from the GPU.
static void BackSub(PluginState *s) {
  // solve "bottom up"
  // These variables are volatile because (I believe) I was hitting a bug with
  // how the hcc compiler was optimizing the loop. If the code works (on HCC)
  // with the "volatile" removed, then it's probably safe to leave it out.
  volatile int i, j;
  int Size;
  Size = s->Size;
  for (i = 0; i < Size; i++) {
    s->finalVec[Size - i - 1] = s->b[Size - i - 1];
    for (j = 0; j < i; j++) {
      s->finalVec[Size - i - 1] -=
        s->a[Size * (Size - i - 1) + (Size - j - 1)] *
        s->finalVec[Size - j - 1];
    }
    s->finalVec[Size - i - 1] = s->finalVec[Size - i - 1] /
      s->a[Size * (Size - i - 1) + (Size - i - 1)];
  }
}

static int CopyOut(void *data, TimingInformation *times) {
  PluginState *s = (PluginState *) data;
  size_t size = s->Size * s->Size * sizeof(float);
  if (!CheckHIPError(hipMemcpyAsync(s->m, s->m_device, size,
    hipMemcpyDeviceToHost, s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipMemcpyAsync(s->a, s->a_device, size,
    hipMemcpyDeviceToHost, s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipMemcpyAsync(s->b, s->b_device, s->Size * sizeof(float),
    hipMemcpyDeviceToHost, s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
  BackSub(s);
  times->kernel_count = 2 * (s->Size - 1);
  times->kernel_times = s->kernel_times;
  times->resulting_data_size = s->Size * sizeof(float);
  times->resulting_data = s->finalVec;
  return 1;
}

static const char* GetName(void) {
  return "Gaussian (Rodinia)";
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

