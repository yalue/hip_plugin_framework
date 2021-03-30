// This file defines a shared library that should be invoked in a child process
// to obtain device information.  The reason for jumping through so many hoops
// is that HIP tried to initialize itself at *load* time, but you can't re-
// initialize HIP in a child process. So, the host process can't directly
// initialize HIP because then the plugins would fail when they are loaded.
// So, instead, the host forks a child process that loads a HIP shared library,
// then communicates back to the parent process via shared memory.
#include <stdio.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include "plugin_interface.h"

extern "C" {
int GetDeviceInformation(int device_id, DeviceInformation *info);
}

// This macro takes a hipError_t value and prints the error value and returns 0
// if the value is not hipSuccess. Returns 1 otherwise.
#define CheckHIPError(val) (InternalHIPErrorCheck((val), #val, __FILE__, __LINE__))

static int InternalHIPErrorCheck(hipError_t result, const char *fn,
    const char *file, int line) {
  if (result == hipSuccess) return 1;
  printf("HIP error %d: %s. In %s, line %d (%s)\n", (int) result,
    hipGetErrorString(result), file, line, fn);
  return 0;
}

// A very simple kernel to read the current GPU clock.
__global__ void GetCurrentClock(uint64_t *clock) {
  *clock = clock64();
}

// Just does a bunch of spinning to hopefully bring the GPU out of a low-power
// state. The dummy argument should be NULL, and is there only to prevent
// optimizations.
__global__ void SpinKernel(uint64_t loops, uint64_t *dummy) {
  uint64_t accumulator = 0;
  while (loops != 0) {
    accumulator += loops % accumulator;
    loops--;
  }
  if (dummy) *dummy = accumulator;
}

// Fills in the DeviceInformation struct. Returns 0 on error.
int GetDeviceInformation(int device_id, DeviceInformation *info) {
  int name_length;
  uint64_t *device_clock = NULL;
  uint64_t device_clock_host[2];
  hipDeviceProp_t properties;
  if (!CheckHIPError(hipSetDevice(device_id))) return 0;

  // First, read basic device information.
  if (!CheckHIPError(hipGetDeviceProperties(&properties, device_id))) return 0;
  info->compute_unit_count = properties.multiProcessorCount;
  info->warp_size = properties.warpSize;
  info->threads_per_compute_unit = properties.maxThreadsPerMultiProcessor;
  info->gcn_architecture = properties.gcnArch;
  name_length = sizeof(info->device_name);
  if (sizeof(properties.name) < name_length) {
    name_length = sizeof(properties.name);
  }
  memcpy(info->device_name, properties.name, name_length);
  if (!CheckHIPError(hipMalloc(&device_clock, sizeof(uint64_t) * 2))) return 0;

  // Next, we'll do a bunch of stuff to get a base GPU clock and GPU clock rate
  // (since the value returned by hipGetDeviceProperties usually isn't that
  // close to the actual rate at runtime. We'll start by "warming up" the
  // clock-reading kernel and running the spin kernel to hopefully bring the
  // GPU out of a low-power state.
  hipLaunchKernelGGL(GetCurrentClock, 1, 1, 0, 0, device_clock);
  hipLaunchKernelGGL(SpinKernel, 128, 64, 0, 0, 100000L, (uint64_t *) NULL);
  if (!CheckHIPError(hipDeviceSynchronize())) {
    hipFree(device_clock);
    return 0;
  }

  // Now, we'll launch the two device kernels, one second apart.
  hipLaunchKernelGGL(GetCurrentClock, 1, 1, 0, 0, device_clock);
  usleep(1000000);
  hipLaunchKernelGGL(GetCurrentClock, 1, 1, 0, 0, device_clock + 1);
  if (!CheckHIPError(hipDeviceSynchronize())) {
    hipFree(device_clock);
    return 0;
  }

  // Now, copy out the clock readings.
  if (!CheckHIPError(hipMemcpy(device_clock_host, device_clock,
    sizeof(uint64_t) * 2, hipMemcpyDeviceToHost))) {
    hipFree(device_clock);
    return 0;
  }
  hipFree(device_clock);
  info->starting_clock = device_clock_host[1];
  info->gpu_clocks_per_second = device_clock_host[1] - device_clock_host[0];
  return 1;
}
