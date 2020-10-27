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
  int i;
  int all_set = 1;
  memset(&to_create, 0, sizeof(to_create));

  // If the mask is all 1's, then don't use hipExtStreamCreateWithCUMask. It
  // has performance implications, as it will always get a new underlying HSA
  // queue. Instead, use the default hipStreamCreate.
  for (i = 0; i < mask_count; i++) {
    if (mask[i] != 0xffffffff) {
      all_set = 0;
      break;
    }
  }
  if (all_set) {
    result = hipStreamCreate(&to_create);
    *stream = to_create;
    return result;
  }

  // We have a non-trivial CU mask, so create the stream with a CU mask if
  // we're using a platform that supports it.
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
