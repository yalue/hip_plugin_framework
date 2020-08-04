# This script runs a series of short tests to build a scatterplot of memory
# copy times vs. buffer size.

import argparse
import json
import subprocess

def generate_config(buffer_size):
    """ Returns a JSON string containing a config. Requires the size, in bytes,
    of the buffer to be copied. """
    plugin_config = {
        "label": str(buffer_size),
        "log_name": "results/memory_copy_bytes_%d.json" % (buffer_size,),
        "filename": "./bin/memory_copy.so",
        "thread_count": 512,
        "block_count": 512,
        "additional_info": {
            "buffer_size": buffer_size,
            "copy_subdivisions": 1,
            "sync_every_copy": False
        }
    }
    overall_config = {
        "name": "Memory copy time vs. buffer size",
        "max_iterations": 100,
        "max_time": 0,
        "gpu_device_id": 0,
        "pin_cpus": True,
        "do_warmup": True,
        "omit_block_times": False,
        "use_processes": False,
        "plugins": [plugin_config]
    }
    return json.dumps(overall_config)

def run_process(buffer_size):
    """ Starts the process that will run the plugin with the given buffer
    size. """
    config = generate_config(buffer_size)
    print("Starting test with buffer size %f MB." %
        (float(buffer_size) / (1024.0 * 1024.0),))
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config)

if __name__ == "__main__":
    for i in range(21):
        buffer_size = 1024 * (2 ** i)
        run_process(buffer_size)

