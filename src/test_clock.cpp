// This is a simple command-line utility for printing a list of HIP devices
// and their IDs.
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

// This macro takes a hipError_t value and prints the error value and returns 0
// if the value is not hipSuccess. Returns 1 otherwise.
#define CheckHIPError(val) (InternalHIPErrorCheck((val), #val, __FILE__, __LINE__))

static int InternalHIPErrorCheck(hipError_t result, const char *fn,
    const char *file, int line) {
  if (result == hipSuccess) return 1;
  printf("HIP error %d: %s. In %s, line %d (%s)\n", (int) result,
    hipGetErrorString(result), file, line, fn);
  exit(1);
}

// The kernel for spinning a certain number of iterations. The dummy argument
// should be NULL, it exists to prevent optimizing out the loop.
__global__ void ReadClockKernel(uint64_t *result) {
  *result = clock64();
}

int main(int argc, char **argv) {
  int device = 0;
  uint64_t *device_clock = NULL;
  uint64_t *host_clock = NULL;
  int i;
  if (argc != 2) {
    printf("Usage: %s <device number>\n", argv[0]);
    return 1;
  }
  device = atoi(argv[1]);
  printf("Running on GPU # %d\n", device);
  CheckHIPError(hipSetDevice(device));
  CheckHIPError(hipMalloc(&device_clock, sizeof(uint64_t)));
  CheckHIPError(hipHostMalloc(&host_clock, sizeof(uint64_t)));
  for (i = 0; i < 3; i++) {
    hipLaunchKernelGGL(ReadClockKernel, 1, 1, 0, 0, device_clock);
    CheckHIPError(hipDeviceSynchronize());
    CheckHIPError(hipMemcpy(host_clock, device_clock, sizeof(uint64_t),
      hipMemcpyDeviceToHost));
    CheckHIPError(hipDeviceSynchronize());
    printf("Clock reading %d: %llu\n", i, (unsigned long long) *host_clock);
  }
  hipHostFree(host_clock);
  hipFree(device_clock);
  return 0;
}
