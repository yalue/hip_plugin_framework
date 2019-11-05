///
/// @file    common.h
/// @author  Martin Jirman (207962@mail.muni.cz)
/// @brief   Common stuff for all CUDA dwt functions.
/// @date    2011-01-20 14:19
///
/// Copyright (c) 2011 Martin Jirman
/// All rights reserved.
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions are met:
///
///     * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
///     * Redistributions in binary form must reproduce the above copyright
///       notice, this list of conditions and the following disclaimer in the
///       documentation and/or other materials provided with the distribution.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
/// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
/// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
/// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
/// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
/// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
/// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
/// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
/// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
/// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.
///

#ifndef DWT_COMMON_H
#define DWT_COMMON_H

#include <algorithm>
#include <cstdio>
#include <vector>

#include "hip/hip_runtime.h"

// compile time minimum macro
#define CTMIN(a, b) (((a) < (b)) ? (a) : (b))

namespace dwt_cuda {

/// Divide and round up.
template <typename T>
__device__ __host__ inline T divRndUp(const T& n, const T& d) {
  return (n / d) + ((n % d) ? 1 : 0);
}

// 5/3 forward DWT lifting schema coefficients
const float forward53Predict = -0.5f;  /// forward 5/3 predict
const float forward53Update = 0.25f;   /// forward 5/3 update

/// Functor which adds scaled sum of neighbors to given central pixel.
struct AddScaledSum {
  const float scale;  // scale of neighbors
  __device__ AddScaledSum(const float scale) : scale(scale) {}
  __device__ void operator()(const float p, float& c, const float n) const {
    c += scale * (p + n);
  }
};

/// Returns index ranging from 0 to num threads, such that first half
/// of threads get even indices and others get odd indices. Each thread
/// gets different index.
/// Example: (for 8 threads)   threadIdx.x:   0  1  2  3  4  5  6  7
///                              parityIdx:   0  2  4  6  1  3  5  7
/// @tparam THREADS  total count of participating threads
/// @return parity-separated index of thread
template <int THREADS>
__device__ inline int parityIdx() {
  return (threadIdx.x * 2) - (THREADS - 1) * (threadIdx.x / (THREADS / 2));
}

/// size of shared memory
const int SHM_SIZE = 16 * 1024;

}  // end of namespace dwt_cuda

#endif  // DWT_COMMON_CUDA_H
