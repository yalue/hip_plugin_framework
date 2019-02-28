#include <stdio.h>
#include <hip/hip_runtime.h>
#include "plugin_hip_utilities.h"

int InternalHIPErrorCheck(hipError_t result, const char *fn, const char *file,
    int line) {
  if (result == hipSuccess) return 1;
  printf("HIP error %d: %s. In %s, line %d (%s)\n", (int) result,
    hipGetErrorString(result), file, line, fn);
  return 0;
}

hipError_t CreateHIPStreamWithMask(hipStream_t *stream, uint64_t *mask,
    int mask_count) {
  hipStream_t to_create;
  hipError_t status = hipSuccess;
  if (mask_count < 0) {
    printf("Internal error creating HIP stream: Invalid mask count.\n");
    return hipErrorInvalidValue;
  }
  status = hipStreamCreate(&to_create);
  if (status != hipSuccess) return status;
  if (mask_count == 0) return hipSuccess;
  status = hipStreamSetComputeUnitMask(to_create, mask[0]);
  if (status != hipSuccess) {
    // Wrap this in an error check to print the message if more errors occur.
    CheckHIPError(hipStreamDestroy(to_create));
    return status;
  }
  *stream = to_create;
  return hipSuccess;
}

hipError_t GetMemoryClockRate(int device_id, int *memory_clock_rate) {
  hipDeviceProp_t properties;
  hipError_t result = hipGetDeviceProperties(&properties, device_id);
  if (result != hipSuccess) return result;
  *memory_clock_rate = properties.memoryClockRate;
  return hipSuccess;
}

