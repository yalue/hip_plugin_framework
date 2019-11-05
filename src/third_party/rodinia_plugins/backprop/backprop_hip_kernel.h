#ifndef BACKPROP_HIP_KERNEL_H
#define BACKPROP_HIP_KERNEL_H
#include <hip/hip_runtime.h>
#include "backprop.h"
#include "backprop_state.h"

__global__ void bpnn_layerforward_CUDA(float *input_cuda,
    float *output_hidden_cuda, float *input_hidden_cuda,
    float *hidden_partial_sum, int in, int hid, uint64_t *block_times);

__global__ void bpnn_adjust_weights_cuda(float *delta, int hid, float *ly,
    int in, float *w, float *oldw, uint64_t *block_times);

#endif  // BACKPROP_HIP_KERNEL_H

