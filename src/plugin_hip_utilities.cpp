#include <stdio.h>
#include <string.h>
#include <hip/hip_runtime.h>
#include "plugin_hip_utilities.h"

int InternalHIPErrorCheck(hipError_t result, const char *fn, const char *file,
    int line) {
  if (result == hipSuccess) return 1;
  printf("HIP error %d: %s. In %s, line %d (%s)\n", (int) result,
    hipGetErrorString(result), file, line, fn);
  return 0;
}



hipError_t CreateHIPStreamWithMask(hipStream_t *stream, uint32_t *mask,
    int mask_count) {
  hipStream_t to_create;
  hipError_t result;
  memset(&to_create, 0, sizeof(to_create));
#ifdef __HIP__
  // This should only run under HIP-clang under ROCm 3.6 or later.
  result = hipExtStreamCreateWithCUMask(&to_create, mask_count, mask);

#else
  // nvcc or versions other than HIP-clang won't support the necessary API.
  printf("Warning: Setting a CU mask isn't supported in this HIP version.\n");
  result = hipStreamCreate(&to_create);
#endif
  *stream = to_create;
  return result;
}
