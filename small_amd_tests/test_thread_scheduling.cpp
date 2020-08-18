// This file is a quick experiment that was intended to help me examine thread
// vs. block scheduling on AMD GPUs. It creates a JSON file:
// "thread_results.json".
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime.h>

#define SPIN_ITERATIONS (10000)

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

// Just wraps the host and device times buffers.
typedef struct {
  uint64_t *start_host;
  uint64_t *start_device;
  uint64_t *end_host;
  uint64_t *end_device;
} TimeResults;

// The kernel for spinning a certain number of iterations. The dummy argument
// should be NULL, it exists to prevent optimizing out the loop.
__global__ void CounterSpinKernel(uint64_t max_iterations, uint64_t *dummy,
    uint64_t *start_times, uint64_t *end_times) {
  uint64_t start_clock = clock64();
  uint64_t i, accumulator;
  uint64_t thread_index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  start_times[thread_index] = start_clock;
  accumulator = 0;
  for (i = 0; i < max_iterations; i++) {
    accumulator += i % hipBlockIdx_x;
  }
  if (dummy) *dummy = accumulator;
  end_times[thread_index] = clock64();
}

// Allocates memory to hold the thread start and end times on the host and
// device.
static void InitializeTimeResults(TimeResults *r, uint64_t total_threads) {
  size_t size = total_threads * sizeof(uint64_t);
  printf("Required memory: %f MB on CPU and GPU.\n",
    2.0 * (((double) size) / (1024.0 * 1024.0)));
  CheckHIPError(hipHostMalloc(&r->start_host, size));
  CheckHIPError(hipHostMalloc(&r->end_host, size));
  CheckHIPError(hipMalloc(&r->start_device, size));
  CheckHIPError(hipMemset(r->start_device, 0, size));
  CheckHIPError(hipMalloc(&r->end_device, size));
  CheckHIPError(hipMemset(r->end_device, 0, size));
  CheckHIPError(hipDeviceSynchronize());
}

// Copies times from the device to the host.
static void CopyResultsOut(TimeResults *r, uint64_t total_threads) {
  size_t size = total_threads * sizeof(uint64_t);
  CheckHIPError(hipMemcpy(r->start_host, r->start_device, size,
    hipMemcpyDeviceToHost));
  CheckHIPError(hipMemcpy(r->end_host, r->end_device, size,
    hipMemcpyDeviceToHost));
  CheckHIPError(hipDeviceSynchronize());
}

static void WriteResultsFile(TimeResults *r, uint64_t block_count,
    uint64_t thread_count) {
  uint64_t thread_index, block, thread, tmp;
  FILE *f = fopen("./thread_results.json", "wb");
  if (!f) {
    printf("Failed opening thread_results.json: %s\n", strerror(errno));
    return;
  }
  // I'm not going to check fprintf for errors, bad practice, I know... The
  // JSON is a single object containing a "blocks" array. Each "block" is an
  // object containing a start_times and end_times array, and thread i in the
  // block has its start and end time at index i in the respective array.
  fprintf(f, "{\n \"blocks\": [\n");

  for (block = 0; block < block_count; block++) {
    fprintf(f, "{\n \"start_times\": [\n");
    for (thread = 0; thread < thread_count; thread++) {
      thread_index = block * thread_count + thread;
      tmp = r->start_host[thread_index];
      fprintf(f, "%llu%s\n", (unsigned long long) tmp,
        thread != (thread_count - 1) ? "," : "");
    }
    fprintf(f, "],\n \"end_times\": [\n");
    for (thread = 0; thread < thread_count; thread++) {
      thread_index = block * thread_count + thread;
      tmp = r->end_host[thread_index];
      fprintf(f, "%llu%s\n", (unsigned long long) tmp,
        thread != (thread_count - 1) ? "," : "");
    }
    fprintf(f, "]\n}%s\n", block != (block_count - 1) ? "," : "");
  }
  fprintf(f, "]\n}\n");
  fclose(f);
}

static void FreeResults(TimeResults *r) {
  CheckHIPError(hipHostFree(r->start_host));
  CheckHIPError(hipHostFree(r->end_host));
  CheckHIPError(hipFree(r->start_device));
  CheckHIPError(hipFree(r->end_device));
  memset(r, 0, sizeof(*r));
}

int main(int argc, char **argv) {
  uint64_t thread_count, block_count, total_threads;
  TimeResults r;
  if (argc != 3) {
    printf("Usage: %s <# of threads per block> <# of blocks>\n", argv[0]);
    return 1;
  }
  thread_count = strtol(argv[1], NULL, 0);
  // We want the number of threads per block to be evenly divisible by 32.
  if ((thread_count <= 0) || (thread_count > 1024) || (thread_count % 32)) {
    printf("Bad # of threads per block: %s\n", argv[1]);
    return 1;
  }
  block_count = strtol(argv[2], NULL, 0);
  if (block_count <= 0) {
    printf("Bad block count: %s\n", argv[2]);
    return 1;
  }
  total_threads = thread_count * block_count;
  InitializeTimeResults(&r, total_threads);
  hipLaunchKernelGGL(CounterSpinKernel, block_count, thread_count, 0, 0,
    SPIN_ITERATIONS, (uint64_t *) NULL, r.start_device, r.end_device);
  CheckHIPError(hipDeviceSynchronize());
  CopyResultsOut(&r, total_threads);
  WriteResultsFile(&r, block_count, thread_count);
  FreeResults(&r);
  return 0;
}

