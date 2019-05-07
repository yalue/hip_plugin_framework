# This script tests the overhead of creating additional threads--basically it
# should let us see how many threads can actually run concurrently on a single
# compute unit.
import json
import subprocess

def generate_config(thread_count):
    plugin_config = {
        "label": str(thread_count),
        "log_name": "results/%d_threads_only.json" % (thread_count,),
        "filename": "./bin/counter_spin.so",
        "thread_count": thread_count,
        "block_count": 1,
        "additional_info": 100000
    }
    overall_config = {
        "name": "Thread count vs. performance",
        "max_iterations": 5,
        "max_time": 0,
        "gpu_device_id": 1,
        "pin_cpus": True,
        "plugins": [plugin_config]
    }
    return json.dumps(overall_config)

def run_process(thread_count):
    config = generate_config(thread_count)
    print "Starting test with %d threads." % (thread_count,)
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config)

max_thread_count = 1024
for thread in range(max_thread_count):
    run_process(thread + 1)

