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

// NOTES ON CHANGES FROM THE ORIGINAL dwt2d BENCHMARK:
//
// - This plugin always generates an in-memory RGB bitmap. The original read
//   from a .bmp file (but I think it reads it wrong--it includes the BMP
//   header in the pixel data).
// - The original benchmark was able to do a forward or reverse transform. This
//   version can only do a forward transform. The original Rodinia benchmark's
//   "run" file only included forward transforms.
// - The original benchmark could do either 9/7 or 5/3 transforms, but this
//   version only does 5/3 transforms (once again, based on the arbitrary
//   reason that the "run" file only specified 5/3 transforms.


#include <assert.h>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include "plugin_hip_utilities.h"
#include "plugin_interface.h"
#include "plugin_utilities.h"

#include "common.h"
#include "components.h"
#include "dwt.h"
#include "dwt_plugin_state.h"

// To simplify the original benchmark, we'll always use a square image. This
// size of 1024 was from one of the original tests included in Rodinia 3.1.
#define IMAGE_WIDTH (1024)

// This value was the default in the original benchmark.
#define DWT_LEVELS (3)

// We only care about 3-channel RGB images here.
#define COLOR_CHANNELS (3)

static void Cleanup(void *data) {
  int i;
  PluginState *s = (PluginState *) data;
  // First free the dwt struct and the original image
  if (s->d) {
    hipHostFree(s->d->srcImg);
  }
  free(s->d);
  // Next, free all the straightforward stuff
  hipFree(s->srcImg_device);
  hipFree(s->c_r_out);
  hipFree(s->c_g_out);
  hipFree(s->c_b_out);
  hipFree(s->backup);
  hipFree(s->c_r);
  hipFree(s->c_g);
  hipFree(s->c_b);
  free(s->kernel_times);
  // Free the lists of block times for each kernel, but only if the lists of
  // pointers were successfully allocated.
  if (s->device_block_times) {
    for (i = 0; i < (DWT_LEVELS + 1); i++) {
      hipFree(s->device_block_times[i]);
    }
    free(s->device_block_times);
  }
  if (s->host_block_times) {
    for (i = 0; i < (DWT_LEVELS + 1); i++) {
      hipHostFree(s->host_block_times[i]);
    }
    free(s->host_block_times);
  }
  // Destroy the stream if it was successfully created.
  if (s->stream_created) {
    CheckHIPError(hipStreamDestroy(s->stream));
  }
  // Finally, free the PluginState struct itself.
  free(s);
}

// Fills in the buffer with an arbitrary pattern of RGB pixels. The buffer must
// be capable of holding a square 3-channel RGB image with the given width.
static void GenerateRGBData(uint8_t *buffer, int width) {
  int x, y, i, w, h;
  uint8_t r, g, b;
  double brightness;
  r = 0;
  g = 0;
  b = 0;
  brightness = 1.0;
  i = 0;
  w = width;
  h = width;
  // Creates an RGB image with an arbitrary pattern, just some bands of color
  // that get progressively darker with descending rows.
  for (y = 0; y < h; y++) {
    brightness = ((double) (h - y)) / ((double) h);
    for (x = 0; x < w; x++) {
      buffer[i] = (uint8_t) ((double) r * brightness);
      i++;
      buffer[i] = (uint8_t) ((double) g * brightness);
      i++;
      buffer[i] = (uint8_t) ((double) b * brightness);
      i++;
      r += 11;
      g += 3;
      b += 5;
    }
  }
}

// Allocates most of the memory used by the plugin. Returns 0 on error. Assumes
// s->d has already been allocated and its integer values set.
static int AllocateMemory(PluginState *s) {
  int component_size, i, j, tmp, width, height;
  uint64_t *tmp_block_times = NULL;
  KernelTimes *k = NULL;
  struct dwt *d = s->d;
  if (!CheckHIPError(hipHostMalloc(&(d->srcImg), IMAGE_WIDTH * IMAGE_WIDTH *
    COLOR_CHANNELS))) {
    return 0;
  }
  GenerateRGBData(d->srcImg, IMAGE_WIDTH);
  if (!CheckHIPError(hipMalloc(&s->srcImg_device, d->pixWidth * d->pixHeight *
    COLOR_CHANNELS))) {
    return 0;
  }
  component_size = d->pixWidth * d->pixHeight * sizeof(int);
  if (!CheckHIPError(hipMalloc(&s->c_r_out, component_size))) return 0;
  if (!CheckHIPError(hipMalloc(&s->c_g_out, component_size))) return 0;
  if (!CheckHIPError(hipMalloc(&s->c_b_out, component_size))) return 0;
  if (!CheckHIPError(hipMalloc(&s->backup, component_size))) return 0;
  if (!CheckHIPError(hipMalloc(&s->c_r, component_size))) return 0;
  if (!CheckHIPError(hipMalloc(&s->c_g, component_size))) return 0;
  if (!CheckHIPError(hipMalloc(&s->c_b, component_size))) return 0;
  // We launch DWT_LEVELS kernels for each color channel, plus a single kernel
  // for separating the colors into channels.
  tmp = (DWT_LEVELS * COLOR_CHANNELS) + 1;
  s->kernel_times = (KernelTimes *) calloc(tmp, sizeof(KernelTimes));
  if (!s->kernel_times) {
    printf("Failed allocating kernel times list.\n");
    return 0;
  }
  // We also need to allocate pointers to block times for each of these
  // kernels.
  s->host_block_times = (uint64_t **) calloc(tmp, sizeof(uint64_t *));
  if (!s->host_block_times) {
    printf("Failed allocating host block time pointers.\n");
    return 0;
  }
  s->device_block_times = (uint64_t **) calloc(tmp, sizeof(uint64_t *));
  if (!s->device_block_times) {
    printf("Failed allocating device block time pointers.\n");
    return 0;
  }
  // We'll go ahead and figure out the block and thread counts for each kernel
  // here, since we'll need to do so to allocate the block_times buffers
  // anyway. The first kernel is color-component separation.
  s->kernel_times[0].kernel_name = "c_CopySrcToComponents";
  s->kernel_times[0].thread_count = COMPONENTS_THREADS;
  tmp = DIVANDRND(d->pixWidth * d->pixHeight, COMPONENTS_THREADS);
  s->kernel_times[0].block_count = tmp;
  // Now we'll need to do the fdwt53 kernel, which may use a different number
  // of threads and blocks for each invocation. We'll just duplicate the
  // relevant part of the thread and block count computations from
  // fdwt53.cpp here.
  for (j = 0; j < COLOR_CHANNELS; j++) {
    width = d->pixWidth;
    height = d->pixHeight;
    for (i = 0; i < DWT_LEVELS; i++) {
      k = s->kernel_times + (j * DWT_LEVELS) + i + 1;
      k->kernel_name = "fdwt53Kernel";
      if (width >= 960) {
        k->thread_count = 192;
      } else if (width >= 480) {
        k->thread_count = 128;
      } else {
        k->thread_count = 64;
      }
      // First calculate the Y dimension of the grid.
      tmp = DIVANDRND(height, 8 * DIVANDRND(height, 15 * 8));
      // Now, multiply it by the X dimension of the grid.
      tmp *= DIVANDRND(width, k->thread_count);
      k->block_count = tmp;
      // The actual algorithm performs a recursive invocation with these values
      // halved. See fdwt53.cpp, function fdwt53(...).
      width = DIVANDRND(width, 2);
      height = DIVANDRND(height, 2);
    }
  }
  // FINALLY we can use the information we calculated above to allocate the
  // block times buffers for each kernel.
  for (i = 0; i < (COLOR_CHANNELS * DWT_LEVELS + 1); i++) {
    k = s->kernel_times + i;
    if (!CheckHIPError(hipHostMalloc(&tmp_block_times, 2 * k->block_count *
      sizeof(uint64_t)))) {
      return 0;
    }
    s->host_block_times[i] = tmp_block_times;
    k->block_times = tmp_block_times;
    tmp_block_times = NULL;
    if (!CheckHIPError(hipMalloc(&tmp_block_times, 2 * k->block_count *
      sizeof(uint64_t)))) {
      return 0;
    }
    s->device_block_times[i] = tmp_block_times;
    tmp_block_times = NULL;
  }

  return 1;
}

static void* Initialize(InitializationParameters *params) {
  PluginState *s = NULL;
  if (!CheckHIPError(hipSetDevice(params->device_id))) {
    return NULL;
  }
  s = (PluginState *) calloc(1, sizeof(*s));
  if (!s) {
    printf("Failed allocating plugin state.\n");
    return NULL;
  }
  s->d = (struct dwt *) calloc(1, sizeof(struct dwt));
  if (!s->d) {
    Cleanup(s);
    return NULL;
  }
  s->d->pixWidth = IMAGE_WIDTH;
  s->d->pixHeight = IMAGE_WIDTH;
  s->d->dwtLvls = 3;
  s->d->components = 3;
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
  // NOTE: All we care about is processDWT<int>(d, 1, 0);
  return s;
}

static int CopyIn(void *data) {
  int i, size;
  PluginState *s = (PluginState *) data;
  KernelTimes *k = NULL;
  // Reset the current_kernel_index back to 0, in preparation for a new
  // iteration.
  s->current_kernel_index = 0;
  // Reset all of the block times to the maximum uint64 value.
  for (i = 0; i < (COLOR_CHANNELS * DWT_LEVELS + 1); i++) {
    k = s->kernel_times + i;
    if (!CheckHIPError(hipMemsetAsync(s->device_block_times[i], 0xff,
      2 * k->block_count * sizeof(uint64_t), s->stream))) {
      return 0;
    }
  }
  size = s->d->pixWidth * s->d->pixHeight * sizeof(int);
  // Reset the device component arrays.
  if (!CheckHIPError(hipMemsetAsync(s->c_r_out, 0, size, s->stream))) return 0;
  if (!CheckHIPError(hipMemsetAsync(s->c_g_out, 0, size, s->stream))) return 0;
  if (!CheckHIPError(hipMemsetAsync(s->c_b_out, 0, size, s->stream))) return 0;
  if (!CheckHIPError(hipMemsetAsync(s->c_r, 0, size, s->stream))) return 0;
  if (!CheckHIPError(hipMemsetAsync(s->c_g, 0, size, s->stream))) return 0;
  if (!CheckHIPError(hipMemsetAsync(s->c_b, 0, size, s->stream))) return 0;
  if (!CheckHIPError(hipMemsetAsync(s->backup, 0, size, s->stream))) return 0;
  // Copy the source image back in.
  if (!CheckHIPError(hipMemcpyAsync(s->srcImg_device, s->d->srcImg,
    s->d->pixWidth * s->d->pixHeight * COLOR_CHANNELS, hipMemcpyHostToDevice,
    s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
  return 1;
}

// MODIFIED:
//  - Apart from moving all of the memory allocation and initialization into
//    Initialize and CopyIn, this function now returns a 0 on error and nonzero
//    otherwise.
template <typename T>
int processDWT(PluginState *s) {
  struct dwt *d = s->d;
  int *c_r_out = s->c_r_out;
  int *c_g_out = s->c_g_out;
  int *c_b_out = s->c_b_out;
  int *backup = s->backup;
  int *c_r = s->c_r;
  int *c_g = s->c_g;
  int *c_b = s->c_b;

  if (!rgbToComponents(c_r, c_g, c_b, d->srcImg, d->pixWidth, d->pixHeight,
    s)) {
    return 0;
  }
  if (!nStage2dDWT(c_r, c_r_out, backup, d->pixWidth, d->pixHeight,
    d->dwtLvls, s)) {
    return 0;
  }
  if (!nStage2dDWT(c_g, c_g_out, backup, d->pixWidth, d->pixHeight,
    d->dwtLvls, s)) {
    return 0;
  }
  if (!nStage2dDWT(c_b, c_b_out, backup, d->pixWidth, d->pixHeight,
    d->dwtLvls, s)) {
    return 0;
  }
  return 1;
}

static int Execute(void *data) {
  PluginState *s = (PluginState *) data;
  if (!processDWT<int>(s)) return 0;
  return 1;
}

static int CopyOut(void *data, TimingInformation *times) {
  PluginState *s = (PluginState *) data;
  KernelTimes *k = NULL;
  int i;
  // For this plugin, we're not going to bother copying output data, but we
  // still need to copy the block times back to the host.
  for (i = 0; i < (COLOR_CHANNELS * DWT_LEVELS + 1); i++) {
    k = s->kernel_times + i;
    if (!CheckHIPError(hipMemcpyAsync(s->host_block_times[i],
      s->device_block_times[i], 2 * k->block_count * sizeof(uint64_t),
      hipMemcpyDeviceToHost, s->stream))) {
      return 0;
    }
  }
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;
  times->kernel_count = COLOR_CHANNELS * DWT_LEVELS + 1;
  times->kernel_times = s->kernel_times;
  times->resulting_data_size = 0;
  times->resulting_data = NULL;
  return 1;
}

static const char* GetName(void) {
  return "DWT2D (Rodinia)";
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

