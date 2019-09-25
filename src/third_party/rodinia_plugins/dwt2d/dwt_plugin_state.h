#ifndef DWT2D_PLUGIN_STATE_H
#define DWT2D_PLUGIN_STATE_H
// This file contains definitions shared between multiple files in the plugin.

#include <stdint.h>
#include <hip/hip_runtime.h>
#include "plugin_interface.h"

struct dwt {
  unsigned char *srcImg;
  int pixWidth;
  int pixHeight;
  int components;
  int dwtLvls;
};

typedef struct {
  hipStream_t stream;
  int stream_created;
  KernelTimes *kernel_times;
  // Arrays of pointers to buffers that are meant to hold device and host block
  // times.
  uint64_t **host_block_times;
  uint64_t **device_block_times;
  struct dwt *d;
  // Other memory allocations used by the original benchmark.
  unsigned char *srcImg_device;
  int *c_r_out;
  int *c_g_out;
  int *c_b_out;
  int *backup;
  int *c_r;
  int *c_g;
  int *c_b;
  // Used by the plugin to keep track of where to record the next kernel's
  // times. Resest during CopyIn.
  int current_kernel_index;
} PluginState;

#endif  // DWT2D_PLUGIN_STATE_H

