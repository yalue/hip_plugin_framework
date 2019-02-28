# This script tests performance when compute units are shared between two
# tasks.
import json
import subprocess

def get_cu_mask(active_count, total_count):
    """ Returns a compute-unit mask as an array of booleans. """
    if active_count > total_count:
        print "Got bad active and total CU count amounts"
        exit()
    return [True] * active_count + [False] * (total_count - active_count)

def generate_config(cu_count, competitor_cu_count, overlap_amount, total_cus):
    """ Returns a JSON string containing a config. The overlap amount is the
    number of compute units shared with a competing workload. """
    competitor_cu_mask = get_cu_mask(competitor_cu_count, total_cus)
    # The competitor will use the higher-numbered CUs, and the measured task
    # will use the lower-numbered ones by default.
    competitor_cu_mask.reverse()
    cu_mask = get_cu_mask(cu_count, total_cus)
    # Set the amount of overlap by rotating the measured task's CU mask left,
    # causing it to use some of the higher-numbered CUs.
    cu_mask = cu_mask[overlap_amount:] + cu_mask[:overlap_amount]
    task_config = {
        "label": str(overlap_amount),
        "log_name": "results/%d_cu_overlap.json" % (overlap_amount),
        "filename": "./bin/mandelbrot.so",
        "thread_count": 512,
        "block_count": 9999,
        "compute_unit_mask": cu_mask,
        "additional_info": 1000
    }
    competitor_config = {
        "label": "competitor",
        "log_name": "results/cu_overlap_competitor.json",
        "filename": "./bin/mandelbrot.so",
        "thread_count": 512,
        "block_count": 9999,
        "compute_unit_mask": competitor_cu_mask,
        "additional_info": 1000
    }
    overall_config = {
        "name": "Compute Unit Overlap vs. Performance",
        "max_iterations": 100,
        "max_time": 0,
        "gpu_device_id": 0,
        "pin_cpus": True,
        "sync_every_iteration": True,
        "use_processes": True,
        "plugins": [task_config, competitor_config]
    }
    return json.dumps(overall_config)

def run_process(cu_count, competitor_cu_count, overlap_amount, total_cus):
    """ This function starts a process that will run the plugin with the given
    compute unit count. """
    config = generate_config(cu_count, competitor_cu_count,
        overlap_amount, total_cus)
    print "Starting test with overlap of %d CUs." % (overlap_amount)
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config)

# This test was designed for the RX 570, with 32 compute units.
for i in range(17):
    run_process(16, 16, i, 32)

