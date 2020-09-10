# This script runs a series of short tests to build a scatterplot of memory
# copy times vs. buffer size.

import argparse
import json
import subprocess

def generate_competitor_configs(count):
    """ Returns a dict containing plugin configs for the given number of
    competitors. """
    to_return = []
    for i in range(count):
        competitor_config = {
            "log_name": "results/memory_copy_competitor_%d.json" % (i,),
            "filename": "./bin/memory_copy.so",
            "thread_count": 512,
            "block_count": 512,
            "additional_info": {
                "buffer_size": 1024 * 1024 * 1024,
                "copy_subdivisions": 1,
                "sync_every_copy": False
            }
        }
        to_return.append(competitor_config)
    return to_return

def generate_config(buffer_size, competitor_count):
    """ Returns a JSON string containing a config. Requires the size, in bytes,
    of the buffer to be copied. """
    plugin_config = {
        "label": str(buffer_size),
        "log_name": "results/memory_copy_bytes_%d_%dcompetitors.json" % (
            buffer_size, competitor_count),
        "filename": "./bin/memory_copy.so",
        "thread_count": 512,
        "block_count": 512,
        "additional_info": {
            "buffer_size": buffer_size,
            "copy_subdivisions": 1,
            "sync_every_copy": False
        }
    }

    competitors = generate_competitor_configs(competitor_count)
    plugins = []
    plugins.append(plugin_config)
    for c in competitors:
        plugins.append(c)
    config_name = "Memory copy time vs. size, " + str(competitor_count)
    config_name += " competitors"
    overall_config = {
        "name": config_name,
        "max_iterations": 100,
        "max_time": 0,
        "gpu_device_id": 0,
        "pin_cpus": True,
        "do_warmup": True,
        "omit_block_times": False,
        "use_processes": False,
        "plugins": plugins
    }
    return json.dumps(overall_config)

def run_process(buffer_size, competitor_count):
    """ Starts the process that will run the plugin with the given buffer
    size. """
    config = generate_config(buffer_size, competitor_count)
    print("Starting test with buffer size %f MB and %d competitors." %
        (float(buffer_size) / (1024.0 * 1024.0), competitor_count))
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--competitors", default=0, type=int,
        help="The number of competing instances to run.")
    parser.add_argument("--dynamic_subdivisions", dest="dynamic_subdivisions",
        action="store_true", help="If set, this causes copies from both the " +
        "measured task and competitors to be subdivided into 64 MB chunks.")
    args = parser.parse_args()
    if args.competitors < 0:
        print("The number of competitors must be positive.")
        exit(1)

    for i in range(32):
        # Test buffer sizes from 64 MB to 2 GB, in steps of 64 MB.
        buffer_size = 64 * (1024 * 1024) * (i + 1)
        run_process(buffer_size, args.competitors, args.dynamic_subdivisions)

