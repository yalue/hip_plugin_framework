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
#include "dwt_hip/dwt.h"
#include <assert.h>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include "plugin_hip_utilities.h"
#include "plugin_utilities.h"

#include "common.h"
#include "dwt.h"
#include "dwt_hip/common.h"
#include "dwt_plugin_state.h"

inline int fdwt(int *in, int *out, int width, int height, int levels,
    PluginState *s) {
  return dwt_cuda::fdwt53(in, out, width, height, levels, s);
}

template <typename T>
int nStage2dDWT(T *in, T *out, T *backup, int pixWidth, int pixHeight,
                int stages, PluginState *s) {

  /* create backup of input, because each test iteration overwrites it */
  const int size = pixHeight * pixWidth * sizeof(T);
  if (!CheckHIPError(hipMemcpyAsync(backup, in, size, hipMemcpyDeviceToDevice,
    s->stream))) {
    return 0;
  }
  if (!CheckHIPError(hipStreamSynchronize(s->stream))) return 0;

  /* Measure time of individual levels. */
  if (!fdwt(in, out, pixWidth, pixHeight, stages, s)) return 0;

  return 1;
}
template int nStage2dDWT<int>(int *, int *, int *, int, int, int,
    PluginState *);

