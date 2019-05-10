# This script tests performance degradation when the GPU is filled with an
# increasing amount of random memory traffic.
import json
import subprocess

def get_cu_mask(active_count, total_count):
    """ Returns a compute-unit mask as an array of booleans. """
    if (active_count > total_count):
        print "Got bad active and total CU count amounts"
        exit()
    return [True] * active_count + [False] * (total_count - active_count)

def generate_config(competitor_cu_count, measure_random_walk):
    """ Returns a JSON string containing a config. The config will contain
    one block's worth of the "measured" task running on a single SM, with the
    rest of the GPU filled with an increasing amount of random_walk work. """
    # The measured task will only use CU 0, so only make it share an SM with
    # the evil workload after all other SMs have been occupied.
    measured_task_mask = get_cu_mask(1, 32)
    competitor_cu_mask = get_cu_mask(competitor_cu_count, 32)
    competitor_cu_mask.reverse()
    measured_task_config = None
    random_walk_buffer_size = 128 * 1024 * 1024
    block_size = 256
    blocks_per_cu = 2048 / block_size
    if measure_random_walk:
        measured_task_config = {
            "label": str(competitor_cu_count),
            "log_name": "results/random_walk_vs_%d_cu_random_walk.json" % (
                competitor_cu_count,),
            "filename": "./bin/random_walk.so",
            "thread_count": block_size,
            "block_count": 1,
            "compute_unit_mask": measured_task_mask,
            "additional_info": {
                "iterations": 50000,
                "buffer_size": random_walk_buffer_size
            }
        }
    else:
        measured_task_config = {
            "label": str(competitor_cu_count),
            "log_name": "results/counter_spin_vs_%d_cu_random_walk.json" % (
                competitor_cu_count,),
            "filename": "./bin/counter_spin.so",
            "thread_count": block_size,
            "block_count": 1,
            "compute_unit_mask": measured_task_mask,
            "additional_info": 100000
        }
    competitor_config = {
        "label": "Competitor random walk task",
        "log_name": "results/random_competitor_%d_cu.json" % (
            competitor_cu_count,),
        "filename": "./bin/random_walk.so",
        "thread_count": block_size,
        "block_count": blocks_per_cu * competitor_cu_count,
        "compute_unit_mask": competitor_cu_mask,
        "additional_info": {
            "iterations": 50000,
            "buffer_size": random_walk_buffer_size
        }
    }
    program_name = "counter_spin"
    if measure_random_walk:
        program_name = "random_walk"
    overall_config = {
        "name": "CU count vs %s performance w/ memory contention" % (
            program_name,),
        "max_iterations": 50,
        "max_time": 0,
        "gpu_device_id": 1,
        "pin_cpus": True,
        "sync_every_iteration": True,
        "use_processes": True,
        "plugins": [measured_task_config, competitor_config]
    }
    return json.dumps(overall_config)

def run_process(competitor_cu_count, measure_random_walk):
    config = generate_config(competitor_cu_count, measure_random_walk)
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config)

for i in range(32):
    print("Testing counter_spin with %d competing CUs" % (i,))
    run_process(i + 1, False)

for i in range(32):
    print("Testing random_walk with %d competing CUs" % (i,))
    run_process(i + 1, True)

