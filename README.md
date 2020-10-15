HIP Plugin Framework
====================

About
-----

This project is based on the [CUDA Scheduling Examiner](https://github.com/yalue/cuda_scheduling_examiner_mirror)
framework, which made it convenient to configure and examine the behavior of
multiple GPU-sharing tasks on NVIDIA.  The HIP plugin framework is
architecturally similar, but has been overhauled to use more intuitive
terminology and to remove any NVIDIA-specific code.  (Which, unfortunately
means that this project is unable to plot which compute units blocks are
assigned to.

Some of this project may require modified versions of AMD's HIP framework or
other ROCm components (as it is intended to support my research). If any such
modifications are needed, they should be included in
[this repository](https://github.com/yalue/rocm_mega_repo).

As of ROCm 3.7, however, this repository should not require any non-standard
HIP functionality.


Basic Compilation and Usage
---------------------------

Compiling this project requires HIP, and `hipcc` must be on your `PATH`. Only
Linux is supported for now, and only AMD GPUs.  (Compilation should also work
on NVIDIA, but is not a priority and may require modifying the makefile.)

To build:

```bash
git clone https://github.com/yalue/hip_plugin_framework
cd hip_plugin_framework
make
```

To test it, run:

```bash
./bin/runner configs/simple.json
```


Rodinia Plugins
---------------

This project contains several plugins that have been ported from the [Rodinia
benchmark suite](http://lava.cs.virginia.edu/Rodinia/download_links.htm). The
code for these plugins are located in the `src/third_party/rodinia_plugins`
directory.  The four plugins, `backprop`, `dwt2d`, `gaussian`, and
`particlefilter` have been modified to clean up code, remove global variables,
and to adhere to the plugin interface required by the project.

Compile these plugins by running `make rodinia_plugins` in the main project
directory. Some sample configs exist for testing these plugins in `configs/`.
For example, running `./bin/runner configs/particlefilter.json` launches a
single instance of the `particlefilter` plugin.


Configuration Files
-------------------

The configuration files specify parameters passed to each plugin along with
some global settings for the entire framework. See `configs/simple.json` for a
minimal working example, using the Mandelbrot-set plugin.

The layout of each configuration file is as follows:

```
{
  "name": <String. Required. The name of this scenario.>,
  "max_iterations": <Number. Required. Default cap on the number of iterations
    for each plugin. 0 = unlimited.>,
  "max_time": <Number. Required. Default cap on the number of number of seconds
    to run each plugin. 0 = unlimited.>,
  "use_processes": <Boolean, defaulting to false. If this is true, each
    plugin is run in a separate process. Normally, they run as threads.>
  "gpu_device_id": <Number. Required. The HIP device to use.>,
  "base_result_directory": <String, defaulting to "./results". This is the
    directory into which individual JSON files from each plugin will be
    written. It must already exist.>,
  "pin_cpus": <Boolean. Optional, defaults to false. If true, attempt to pin
    plugins to CPU cores, evenly distributed across cores. If true, individual
    plugin's cpu_core settings are ignored.>,
  "do_warmup": <Boolean. Optional, defaults to false. If true, the framework
    will run a warmup iteration of each plugin immediately after
    initialization. The times from the warmup iteration will not be included in
    result logs, so this option can be used to make sure code and data is
    brought into the relevant caches, if possible, prior to the first
    iteration.>,
  "sync_every_iteration": <Boolean. Optional, defaults to false. If true,
    iterations of each plugins start only when all plugins have completed their
    previous iteration. By default, each plugin only waits for its own previous
    iteration to complete.>,
  "omit_block_times": <Boolean. Optional, defaults to false. If true, block
    times will not be listed in results JSON files. Instead, the block_times
    field will always be set to an empty array. This can save time and space
    when running experiments where you don't care about block times. They'll
    still be recorded by plugins--just not written to the output file.>,
  "plugins": [
    {
      "filename": <String. Required. The path to the plugin shared library,
        relative to the current working directory.>,
      "log_name": <String. Optional. The filename of the JSON log for this
        particular plugin. If not provided, this plugin's log will be given a
        default name based on its filename, process and thread ID. If this
        doesn't start with '/', it will be relative to base_result_directory.
        This can be /dev/null to throw away logs.>,
      "label:": <String. Optional. A label or name for this specific plugin, to
      be copied to its output file.>,
      "thread_count": <Number. Required, but may be ignored. The number of HIP
        threads that each block of this plugin should use.>,
      "block_count": <Number. Required, but may be ignored. The number of HIP
        blocks this plugin's kernels should use.>,
      "additional_info": <A JSON object of any format. Optional. This can be
        used to pass additional plugin-specific configuration parameters.>,
      "max_iterations": <Number. Optional. If specified, overrides the default
        max_iterations for this plugin alone. 0 = unlimited. If this is
        provided for any plugin, then sync_every_iteration must be false.>,
      "max_time": <Number. Optional. If specified, overrides the default
        max_time for this plugin alone. 0 = unlimited.>,
      "release_time": <Number. Optional. If set, this plugin will sleep for the
      given number of seconds (between initialization and the start of the
      first iteration) before beginning execution.>,
      "cpu_core": <Number. Optional. If specified, and pin_cpus is false, the
        system will attempt to pin this plugin onto the given CPU core.>
      "compute_unit_mask": <Optional. Can be an array of booleans, or a string
        of binary digits, or a hexadecimal string starting with "0x". This will
        be used by the framework to attempt to configure the CUs on which the
        plugin's kernels are allowed to run. If the value is a string, it is
        interpreted as a list of bits, where if bit i is 1, then execution is
        allowed on CU i. If the value is an array of booleans execution is
        allowed on CU i if entry i is true. If this is not provided or too
        short, then unspecified values default to being enabled.>
    },
    {
      <more plugin instances can be listed here>
    }
  ]
}
```

Additionally, configurations support the insertion of comments via the usage of
"comment" keys, which will be ignored at runtime.

Output File Format
------------------

Each plugin, when run, will generate a JSON log file at the location specified
in the configuration. If the plugin did not complete successfully, the JSON
file may be in an invalid state. Times will be recorded as floating-point
numbers of seconds. The format of the log file is:

```
{
  "scenario_name": "<Scenario name>",
  "plugin_name": "<Plugin name>",
  "label": "<This plugin's label, if given in the config>",
  "max_resident_threads": <The maximum number of threads that can be assigned
    to the GPU at a time (from all plugins in the scenario)>,
  "data_size": <Data size>,
  "release_time": <Requested release time in seconds>,
  "PID": <pid>,
  "TID": <The thread ID, if plugins were run as threads>,
  "times": [
    {},
    {
      "cpu_times": [
        <The CPU time before the copy_in function was called>,
        <The CPU time after the copy_out function returned>
      ],
      "copy_in_times": [
        <The CPU time before the copy_in function was called>,
        <The CPU time after the copy_in function returned>
      ],
      "execute_times": [
        <The CPU time when the execute function was called>,
        <The CPU time after the execute function returned>
      ],
      "copy_out_times": [
        <The CPU time when the copy_out function was called>,
        <The CPU time after the copy_out function returned>
      ],
      "cpu_core": <The current CPU core being used>
    },
    {
      "kernel_name": <The name of this particular kernel. May be omitted.>,
      "block_count": <The number of blocks in this kernel invocation.>,
      "thread_count": <The number of threads per block in this invocation.>,
      "shared_memory": <The amount of shared memory used by this kernel.>,
      "kernel_launch_times": [<CPU time immediately before the kernel launch.>,
        <CPU time immediately after kernel launch returned.>,
        <CPU time immediately after hipStreamSynchronize returned. This will
        be set to 0 if hipStreamSynchronize isn't called for this kernel.>],
      "block_times": [<Start time>, <End time>, ..., <This may be empty if the
        plugin doesn't record block times. This contains block start and end
        times in *millions of GPU cycles*, not seconds.>]
    },
    ...
  ]
}
```

Notice that the first entry in the "times" array will be blank and should be
ignored. The times array will contain two types of objects: one will contain
CPU times and one type will apply to kernel times. An object containing CPU
times will contain a `"cpu_times"` key. A single CPU times object will
encompass all kernel times following it, up until another CPU times object.

Creating New Plugins
--------------------

Each plugin must be contained in a shared library and abide by the interface
specified in `src/plugin_interface.h`. In particular, the library must export
a `RegisterPlugin` function, which provides the addresses of further functions
to the framework. Plugins should preferably never use global state and instead
use the `user_data` pointer returned by the initialize function to track all
state. The reason for this is that we want to be able to run multiple instances
of a single plugin at a time--global variables prevent instances of a single
plugin from being independent. Similarly to global variables, plugins should
use a user-created HIP stream in order to avoid unnecessarily blocking each
other by `hipDeviceSynchronize` (or similar) calls.

The most important piece of information that each plugin provides is the
`TimingInformation` struct, which it must fill in during its `copy_out`
function. This struct will contain a list of `KernelTimes` structs, one for
each kernel invocation called during `execute`. Each `KernelTimes` struct will
contain the kernel start and end times and, if possible, individual block start
and end times (we recognize this may be quite obnoxious to add to some plugins,
so `block_times` are treated as quite desirable, but still optional). The
plugin is responsible for ensuring that the buffers provided in the
`TimingInformation` struct remain valid at least until another plugin function
is called. They will not be freed by the caller.

In general, the comments in `plugin_interface.h` provide an explanation for
the actions that every plugin-provided function is expected to carry out. The
`src/mandelbrot.cpp` file contains a well-commented example of a working plugin
implementation.

In addition to `plugin_interface.h`, `plugin_utilities.h` and
`plugin_hip_utilities.h` define a library of utility functions that may be
used by plugins.

Plugins are invoked by the framework as follows:

 1. The shared library file is loaded using the `dlopen()` function, and the
    `RegisterPlugin` function is located using `dlysym()`.

 2. Depending on the configuration, either a new process or new thread will be
    created for each plugin.

 3. In its own thread or process, the plugin's `initialize` function will be
    called, in which the plugin should allocate and initialize all of the local
    state necessary for one instance of itself.

 4. When the plugin begins running, a single "iteration" will consist of the
    plugin's `copy_in`, `execute`, and `copy_out` functions being called, in
    that order.

 5. When enough time has elapsed or the maximum number of iterations has been
    reached, the plugin's `cleanup` function will be called, to allow for the
    plugin to clean up and free its local state.

 6. If any of the plugin's functions, apart from `initialize` return an error,
    the framework will still call the plugin's `cleanup` function, and then
    cease calling further functions from the plugin.

Coding Style
------------

Even though HIP supports C++, contributions to this project should use the C
programming language when possible. C or HIP source code should adhere to the
parts of the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
that apply to the C language.

Scripts should remain in the `scripts/` directory and should be written in
python when possible. For now, there is no explicit style guide for python
scripts apart from trying to maintain a consistent style within each file.

