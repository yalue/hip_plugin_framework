// This file defines some utility functions that may be useful for plugins
// running HIP code.
#ifndef PLUGIN_HIP_UTILITIES_H
#define PLUGIN_HIP_UTILITIES_H
#ifdef __cplusplus
extern "C" {
#endif

#include <hip/hip_runtime.h>

// This macro takes a hipError_t value and prints the error value and returns 0
// if the value is not hipSuccess. Returns 1 otherwise.
#define CheckHIPError(val) (InternalHIPErrorCheck((val), #val, __FILE__, __LINE__))

int InternalHIPErrorCheck(hipError_t result, const char *fn, const char *file,
    int line);

// Creates the HIP stream and sets the compute unit mask. Returns hipSuccess on
// success. The mask_count argument must be set to the number of 64-bit entries
// in the provided mask.
hipError_t CreateHIPStreamWithMask(hipStream_t *stream, uint64_t *mask,
    int mask_count);

// Sets the memory_clock_rate value to the rate reported by the
// hipGetDeviceProperties function. Returns hipSuccess on success. Remember
// that the rate will be specified in kHz.
hipError_t GetMemoryClockRate(int device_id, int *memory_clock_rate);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // PLUGIN_HIP_UTILITIES_H

