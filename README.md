HIP Plugin Framework
====================

About
-----

This project is based on the [CUDA Scheduling Examiner][https://github.com/yalue/cuda_scheduling_examiner_mirror]
framework, which made it convenient to configure and examine the behavior of
multiple GPU-sharing tasks on NVIDIA.  The HIP plugin framework is
architecturally similar, but intended primarily to carry out a similar task on
AMD GPUs, which have the benefit of greater open-source availability.

Since this project is primarily intended to support my own research, some of it
will require modified versions of AMD's HIP framework, and the entire ROCm
infrastructure.

Basic Compilation and Usage
---------------------------

Compiling this project requires HIP, and `hipcc` must be on your `PATH`. Only
Linux is supported for now, and only AMD GPUs.  Specifically modified versions
of HIP (and HIP's dependencies) may be required for some parts of the code.

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

