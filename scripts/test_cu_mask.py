# This script tests performance under various compute unit mask settings.
import json
import subprocess


def generate_config(active_cu_count, total_cu_count):
    """ Returns a JSON string containing a config. The config will use the
    Mandelbrot plugin with basic settings, but varying the compute unit mask.
    """
    cu_mask = [True] * active_cu_count + [False] * (total_cu_count -
        active_cu_count)
    plugin_config = {
        "label": str(active_cu_count),
        "log_name": "results/%d_compute_units.json" % (active_cu_count),
        "filename": "./bin/mandelbrot.so",
        "thread_count": 512,
        "block_count": 9999,
        "compute_unit_mask": cu_mask,
        "additional_info": 1000
    }
    overall_config = {
        "name": "Compute Unit Count vs. Performance",
        "max_iterations": 100,
        "max_time": 0,
        "gpu_device_id": 1,
        "pin_cpus": True,
        "do_warmup": True,
        "plugins": [plugin_config]
    }
    return json.dumps(overall_config)

def run_process(active_cu_count, total_cu_count):
    """ This function starts a process that will run the plugin with the given
    compute unit count. """
    config = generate_config(active_cu_count, total_cu_count)
    print "Starting test with %d/%d CUs." % (active_cu_count, total_cu_count)
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config)

# This test was designed for the RX 570, with 32 compute units.
total_count = 32
for active in range(total_count):
    run_process(active + 1, total_count)

