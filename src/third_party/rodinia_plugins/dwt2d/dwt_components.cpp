/*
 * Copyright (c) 2009, Jiri Matela
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <errno.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <hip/hip_runtime.h>

#include "plugin_hip_utilities.h"
#include "plugin_utilities.h"

#include "common.h"
#include "components.h"
#include "dwt_plugin_state.h"

/* Store 3 RGB float components */
__device__ void storeComponents(float *d_r, float *d_g, float *d_b, float r,
                                float g, float b, int pos) {
  d_r[pos] = (r / 255.0f) - 0.5f;
  d_g[pos] = (g / 255.0f) - 0.5f;
  d_b[pos] = (b / 255.0f) - 0.5f;
}

/* Store 3 RGB intege components */
__device__ void storeComponents(int *d_r, int *d_g, int *d_b, int r, int g,
                                int b, int pos) {
  d_r[pos] = r - 128;
  d_g[pos] = g - 128;
  d_b[pos] = b - 128;
}

/* Store float component */
__device__ void storeComponent(float *d_c, float c, int pos) {
  d_c[pos] = (c / 255.0f) - 0.5f;
}

/* Store integer component */
__device__ void storeComponent(int *d_c, int c, int pos) { d_c[pos] = c - 128; }

/* Copy img src data into three separated component buffers */
template <typename T>
__global__ void c_CopySrcToComponents(T *d_r, T *d_g, T *d_b,
    unsigned char *d_src, int pixels, uint64_t *block_times) {
  uint64_t start_time = clock64();
  if (start_time < block_times[blockIdx.x * 2]) {
    block_times[blockIdx.x * 2] = start_time;
  }
  int x = threadIdx.x;
  int gX = blockDim.x * blockIdx.x;

  __shared__ unsigned char sData[COMPONENTS_THREADS * 3];

  /* Copy data to shared mem by 4bytes
     other checks are not necessary, since
     d_src buffer is aligned to sharedDataSize */
  if ((x * 4) < COMPONENTS_THREADS * 3) {
    float *s = (float *)d_src;
    float *d = (float *)sData;
    d[x] = s[((gX * 3) >> 2) + x];
  }
  __syncthreads();

  T r, g, b;

  int offset = x * 3;
  r = (T)(sData[offset]);
  g = (T)(sData[offset + 1]);
  b = (T)(sData[offset + 2]);

  int globalOutputPosition = gX + x;
  if (globalOutputPosition < pixels) {
    storeComponents(d_r, d_g, d_b, r, g, b, globalOutputPosition);
  }
  block_times[blockIdx.x * 2 + 1] = clock64();
}

/* Separate compoents of 8bit RGB source image */
// MODIFIED: Now takes a PluginState instance and returns a 0 if an error
// occurs.
template <typename T>
int rgbToComponents(T *d_r, T *d_g, T *d_b, unsigned char *src, int width,
                     int height, PluginState *s) {
  // I know this should always be kernel index 0, but I'll do things this way
  // just for consistency with the other kernels in this plugin.
  KernelTimes *k = s->kernel_times + s->current_kernel_index;
  uint64_t *device_block_times = s->device_block_times[
    s->current_kernel_index];
  s->current_kernel_index++;
  unsigned char *d_src = s->srcImg_device;
  int pixels = width * height;
  // aligned to thread block size -- COMPONENTS_THREADS
  int alignedSize = DIVANDRND(width * height, COMPONENTS_THREADS) *
    COMPONENTS_THREADS * 3;

  /* Kernel */
  dim3 threads(COMPONENTS_THREADS);
  dim3 grid(alignedSize / (COMPONENTS_THREADS * 3));
  assert(alignedSize % (COMPONENTS_THREADS * 3) == 0);
  k->kernel_launch_times[0] = CurrentSeconds();
  hipLaunchKernelGGL((c_CopySrcToComponents), grid, threads, 0, s->stream,
                     d_r, d_g, d_b, d_src, pixels, device_block_times);
  k->kernel_launch_times[1] = CurrentSeconds();
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
  k->kernel_launch_times[2] = CurrentSeconds();
  return 1;
}

template int rgbToComponents<int>(int *d_r, int *d_g, int *d_b,
    unsigned char *src, int width, int height, PluginState *s);

