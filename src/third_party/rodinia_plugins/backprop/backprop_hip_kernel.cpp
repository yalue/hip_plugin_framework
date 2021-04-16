#include <hip/hip_runtime.h>
#include "backprop.h"
#include "backprop_state.h"
#include "backprop_hip_kernel.h"

__global__ void bpnn_layerforward_CUDA(float *input_cuda,
    float *output_hidden_cuda, float *input_hidden_cuda,
    float *hidden_partial_sum, int in, int hid, uint64_t *block_times) {
  uint64_t start_time = clock64();
  int block_index = blockIdx.y * gridDim.x + blockIdx.x;
  if (start_time < block_times[block_index * 2]) {
    block_times[block_index * 2] = start_time;
  }
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
  int index_in = HEIGHT * by + ty + 1;
  __shared__ float input_node[HEIGHT];
  __shared__ float weight_matrix[HEIGHT][WIDTH];

  if (tx == 0) input_node[ty] = input_cuda[index_in];
  __syncthreads();
  weight_matrix[ty][tx] = input_hidden_cuda[index];
  __syncthreads();
  weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];
  __syncthreads();

  for (int i = 1; i <= __log2f(HEIGHT); i++) {
    int power_two = __powf(2, i);
    if (ty % power_two == 0) {
      weight_matrix[ty][tx] = weight_matrix[ty][tx] +
        weight_matrix[ty + power_two / 2][tx];
    }
    __syncthreads();
  }
  input_hidden_cuda[index] = weight_matrix[ty][tx];
  __syncthreads();
  if (tx == 0) {
    hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
  }
  block_times[block_index * 2 + 1] = clock64();
}

__global__ void bpnn_adjust_weights_cuda(float *delta, int hid, float *ly,
    int in, float *w, float *oldw, uint64_t *block_times) {
  uint64_t start_time = clock64();
  int block_index = blockIdx.y * gridDim.x + blockIdx.x;
  if (start_time < block_times[block_index * 2]) {
    block_times[block_index * 2] = start_time;
  }
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int index = (hid + 1) * HEIGHT * by + (hid + 1) * ty + tx + 1 + (hid + 1);
  int index_y = HEIGHT * by + ty + 1;
  int index_x = tx + 1;
  w[index] += (ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]);
  oldw[index] = (ETA * delta[index_x] * ly[index_y]) +
    (MOMENTUM * oldw[index]);
  __syncthreads();
  if (ty == 0 && by ==0) {
    w[index_x] += (ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]);
    oldw[index_x] = (ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]);
  }
  block_times[block_index * 2 + 1] = clock64();
}

