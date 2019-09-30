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

int main(int argc, char **argv) {
  hipDeviceProp_t props;
  int count, i;
  count = 0;
  CheckHIPError(hipGetDeviceCount(&count));
  printf("Found %d devices:\n", count);
  for (i = 0; i < count; i++) {
    CheckHIPError(hipGetDeviceProperties(&props, i));
    printf("Device %d: %s, gcnArch %d, PCI Bus %d, compute mode %d\n", i,
      props.name, props.gcnArch, props.pciBusID, props.computeMode);
  }
  return 0;
}
