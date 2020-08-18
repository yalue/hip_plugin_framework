// This file is a quick experiment that was intended to help me examine the
// granularity with which clock64() can be read on AMD GPUs.
// It creates a JSON file: "timer_results.json", containing a single array.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <hip/hip_runtime.h>

#define TIMER_SAMPLES (100000)

// This macro takes a hipError_t value and exits if it isn't equal to
// hipSuccess.
#define CheckHIPError(val) (InternalHIPErrorCheck((val), #val, __FILE__, __LINE__))

static void InternalHIPErrorCheck(hipError_t result, const char *fn,
    const char *file, int line) {
  if (result == hipSuccess) return;
  printf("HIP error %d: %s. In %s, line %d (%s)\n", (int) result,
    hipGetErrorString(result), file, line, fn);
  exit(1);
}

static double CurrentSeconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    printf("Error getting time.\n");
    exit(1);
  }
  return ((double) ts.tv_sec) + (((double) ts.tv_nsec) / 1e9);
}


// The kernel to poll the timer, recording all of the readings in an array. The
// kernel should use one block with one thread.
__global__ void PollTimerKernel(uint64_t *times, uint64_t iterations) {
  for (uint64_t i = 0; i < iterations; i++) {
    times[i] = clock64();
  }
}

int main(int argc, char **argv) {
  double start_time, end_time;
  FILE *output_file = NULL;
  uint64_t *device_times, *host_times;
  size_t times_size = TIMER_SAMPLES * sizeof(uint64_t);
  CheckHIPError(hipHostMalloc(&host_times, times_size));
  CheckHIPError(hipMalloc(&device_times, times_size));

  // Warm-up run.
  hipLaunchKernelGGL(PollTimerKernel, 1, 1, 0, 0, device_times, TIMER_SAMPLES);
  CheckHIPError(hipDeviceSynchronize());

  // Actual run.
  start_time = CurrentSeconds();
  hipLaunchKernelGGL(PollTimerKernel, 1, 1, 0, 0, device_times, TIMER_SAMPLES);
  CheckHIPError(hipDeviceSynchronize());
  end_time = CurrentSeconds();
  printf("Running the kernel took %f seconds.\n", end_time - start_time);
  CheckHIPError(hipMemcpy(host_times, device_times, times_size,
    hipMemcpyDeviceToHost));
  CheckHIPError(hipDeviceSynchronize());
  output_file = fopen("timer_results.json", "wb");
  if (!output_file) {
    printf("Failed opening timer_results.json: %s\n", strerror(errno));
    return 1;
  }
  fprintf(output_file, "[\n");
  for (uint64_t i = 0; i < TIMER_SAMPLES; i++) {
    fprintf(output_file, "  %lu%s\n", (unsigned long) host_times[i],
      i != (TIMER_SAMPLES - 1) ? "," : "");
  }
  fprintf(output_file, "]\n");
  fclose(output_file);
  hipHostFree(host_times);
  hipFree(device_times);
  return 0;
}

