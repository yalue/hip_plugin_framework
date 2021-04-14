// This file defines the interface that each plugin (.so file) must export.
// Plugins may print an error message if they fail. Otherwise, they are not
// expected to write any output on their own.
#ifndef PLUGIN_INTERFACE_H
#define PLUGIN_INTERFACE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// The number of 32-bit values that can comprise the compute unit bitmask.
#define COMPUTE_UNIT_MASK_ENTRIES (4)

// This struct is filled in by the host, and passed to plugins for reference.
// Typically, howerever, plugins should not rely on device-specific
// information.
typedef struct {
  // The name of the device, as reported by HIP.
  char device_name[256];
  // The architecture of the device.
  int gcn_architecture;
  int compute_unit_count;
  int threads_per_compute_unit;
  uint64_t gpu_clocks_per_second;
  int warp_size;
  // This will be set to the GPU's clock at the time the struct is populated.
  // Usually, this will be used to determine a starting clock time.
  uint64_t starting_clock;
} DeviceInformation;

// This struct is used to pass arguments to the plugin's initialize function.
// Each plugin is not required to use every field, but all fields will always
// be set by the caller regardless.
typedef struct {
  DeviceInformation *device_info;
  // The number of threads to be used in each block. This is a 3-dimensional
  // value, with the unused dimensions being set to 1 (similarly to the dim3
  // arguments passed to hipLaunchKernelGGL).
  int block_dim[3];
  // The number of blocks to create when launching a kernel. Like block_dim,
  // unused dimensions will be set to 1.
  int grid_dim[3];
  // Contains an optional user-specified string, which is taken from the
  // additional_info field in the plugin's JSON config. In practice, this
  // is often a JSON object with plugin-specific parameters.
  char *additional_info;
  // This is the GPU device ID to use in the plugin.
  int device_id;
  // This is used as a list of bits to enable or disable GPU compute units.
  // For example, the lowest bit of compute_unit_mask[0] must be 1 in order to
  // use compute unit 0 on the GPU. compute_unit_mask[0] corresponds to compute
  // units 0 through 31, compute_unit_mask[1] to units 32-63, etc. (At the
  // moment, no devices should include 32 * 4 compute units anyway.) By default
  // this will be set entirely to 1s.
  uint32_t compute_unit_mask[COMPUTE_UNIT_MASK_ENTRIES];
} InitializationParameters;

// This holds times and metadata about a single GPU kernel's execution in a
// single iteration of a plugin's execute() function.
typedef struct {
  // This is the name of the GPU kernel. If this is set to NULL, it will be
  // ignored by the framework.
  const char *kernel_name;
  // The total number of threads per block used by this kernel.
  int thread_count;
  // The total number of blocks run by this kernel.
  int block_count;
  // The total amount of shared memory (in bytes) used by this kernel.
  uint32_t shared_memory;
  // This includes the times measured on the CPU, in seconds, immediately
  // before and after the kernel was launched. The third entry is the time
  // after synchronization completes when the kernel is finished executing.
  double kernel_launch_times[3];
  // This includes the start and end time for every block launched on the GPU,
  // in *GPU clock cycles*. Even-numbered positions contain start times, and
  // odd-numbered positions contain end times. Plugins may set this to NULL if
  // they don't provide block times.
  uint64_t *block_times;
} KernelTimes;

// This holds the measurements obtained during a single iteration of the
// plugin's execute phase. Any pointers in this struct must be managed by the
// plugin and remain valid until another plugin function is called by the
// framework.
typedef struct {
  // The number of kernels run by the plugin in this iteration.
  int kernel_count;
  // This must contain one KernelTimes struct per kernel, with a total of
  // kernel_count entries.
  KernelTimes *kernel_times;
  // This buffer may be set to host data resulting from the plugin's execution
  // (e.g. an output image). This should be set to NULL if the plugin doesn't
  // use it. If non-NULL, the plugin must ensure that this pointer remains
  // valid until another plugin function is called by the framework.
  uint64_t resulting_data_size;
  void *resulting_data;
} TimingInformation;

// This function is called to initialize the plugin. It returns a pointer to
// plugin-defined data that will be passed to all subsequent plugin functions.
// It is an error for this function to return NULL.
typedef void* (*InitializeFunction)(InitializationParameters *params);

// This function should be used to copy data into GPU memory. It will be called
// before the plugin's Execute(...) function. It will receive a pointer to the
// plugin-defined data returned by the initialize function. This function must
// return nonzero on success, or 0 on error.
typedef int (*CopyInFunction)(void *data);

// This function should be used to execute GPU code. This receives the pointer
// to the plugin-specific data returned by the initialize function. This must
// return nonzero on success and 0 on error.
typedef int (*ExecuteFunction)(void *data);

// This function is always called after the plugin's execute function, and
// should be used to copy data out from GPU memory. Additionally, this function
// must fill in the TimingInformation struct. Any pointers to buffers in the
// TimingInformation struct must remain valid until another function in the
// plugin is called by the framework. This must return nonzero on success and
// 0 on error.
typedef int (*CopyOutFunction)(void *data, TimingInformation *times);

// This function will be called before the framework exits, and should be used
// to free any buffers or resources used by this plugin. It receives a copy to
// the plugin-specific buffer initially returned by the plugin's initialize
// function.
typedef void (*CleanupFunction)(void *data);

// This function must return a pointer to a constant null-terminated string
// containing the plugin's name.
typedef const char* (*GetNameFunction)(void);

// The plugin must fill in the members of this struct in the RegisterFunctions
// function. None of these functions may be set to NULL.
typedef struct {
  InitializeFunction initialize;
  CopyInFunction copy_in;
  ExecuteFunction execute;
  CopyOutFunction copy_out;
  CleanupFunction cleanup;
  GetNameFunction get_name;
} PluginFunctions;

// Every plugin must export this function, which will be the first thing to be
// called after dlopen(...). The plugin must fill in the functions struct and
// return nonzero on success, or 0 if an error occurs.
extern int RegisterPlugin(PluginFunctions *functions);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // PLUGIN_INTERFACE_H
