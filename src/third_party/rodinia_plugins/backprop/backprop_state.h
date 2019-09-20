// This file contains the shared PluginState definition used by this port of
// the backprop benchmark--basically so that the definition can be used in
// multiple files.
#ifndef BACKPROP_STATE_H
#define BACKPROP_STATE_H

#include <stdlib.h>
#include <hip/hip_runtime_api.h>
#include "plugin_interface.h"
#include "backprop.h"

struct PluginState {
  // Fields used in basically all plugins, see one of the others for details.
  hipStream_t stream;
  int stream_created;
  KernelTimes kernel_times[2];
  struct drand48_data rng;
  int layer_size;
  unsigned int num_blocks;
  BPNN *net;
  // Memory buffers that were previously local to bpnn_train_cuda--now here so
  // they don't need to be re-allocated between iterations.
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
};

// A thread-safe replacement for places where the original benchmark used
// rand().
float RandomFloat(PluginState *s);

#endif  // BACKPROP_STATE_H

