# This script tests performance degradation when the GPU is filled with an
# increasing amount of random memory traffic.
# This script tests matrix multiply performance against a random-walk
# competitor.
import json
import subprocess

def generate_config(se_packed, use_competitor):
    """ Returns a JSON string containing a config. The config will contain
    a measured matrix-multiply task along with a random-walk competitor. The
    competitor and the measured task run on separate partitions of 24 SMs.
    If se_packed is True, then each task will run on separate shader engines.
    Note that this is intended to be used on a GPU like the Radeon VII, with
    at least 48 CUs and 4 shader engines. """
    # The measured task will only use CU 0, so only make it share an SM with
    # the evil workload after all other SMs have been occupied.
    measured_cu_mask = None
    competitor_cu_mask = None
    label_postfix = None
    filename_postfix = None
    # The "full" 50 bit CU mask would be 0x3ffffffffffff, but we won't use two
    # of the CUs so that we'll be able to evenly spread CUs across SEs when
    # se_packed is false.
    if se_packed:
        measured_cu_mask = "0x0000aaaaaaaaaaaa"
        competitor_cu_mask = "0x0000555555555555"
        label_postfix = ", SE-packed"
        filename_postfix = "_packed"
    else:
        measured_cu_mask = "0x0000ffffff000000"
        competitor_cu_mask = "0x0000000000ffffff"
        label_postfix = ", SE-distributed"
        filename_postfix = "_distributed"
    if use_competitor:
        label_postfix += ", vs. random walk"
        filename_postfix += "_vsrandom"
    else:
        label_postfix += ", no competitor"
        filename_postfix += "_isolated"
    measured_config = {
        "label": "MM1024" + label_postfix,
        "log_name": "results/random_walk_24CU" + filename_postfix + ".json",
        "filename": "./bin/matrix_multiply.so",
        "thread_count": [32, 32],
        "block_count": 1,
        "compute_unit_mask": measured_cu_mask,
        "additional_info": {
            "matrix_width": 1024,
            "skip_copy": False
        }
    }
    competitor_config = {
        "label": "Competitor",
        "log_name": "/dev/null",
        "filename": "./bin/random_walk.so",
        "thread_count": 256,
        "block_count": 1000,
        "additional_info": {
            "iterations": 10000,
            "buffer_size": 32 * 1024 * 1024
        }
    }
    plugin_list = [measured_config]
    if use_competitor:
        plugin_list.append(competitor_config)
    overall_config = {
        "name": "Memory interference vs. CU partitioning scheme",
        "max_iterations": 0,
        "max_time": 60,
        "gpu_device_id": 0,
        "pin_cpus": True,
        "sync_every_iteration": True,
        "use_processes": True,
        "plugins": plugin_list
    }
    return bytes(json.dumps(overall_config), 'utf-8')

def run_process(se_packed, use_competitor):
    config = generate_config(se_packed, use_competitor)
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config)

print("Testing with SE-packed")
run_process(True, True)
run_process(True, False)
print("Testing with SE-distributed")
run_process(False, True)
run_process(False, False)

