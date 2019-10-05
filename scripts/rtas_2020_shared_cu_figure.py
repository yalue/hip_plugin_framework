# This script tests the impact of CU cluster on interference.
import json
import subprocess

def task_config(label, log_name, cu_mask):
    """ Returns the config for the measured task, as a dict. """
    task_config_old = {
        "label": label,
        "log_name": log_name,
        "filename": "./bin/particlefilter_naive.so",
        "thread_count": 0,
        "block_count": 0,
        "compute_unit_mask": cu_mask
    }
    task_config = {
        "label": label,
        "log_name": log_name,
        "filename": "./bin/mandelbrot.so",
        "thread_count": 512,
        "block_count": 9999,
        "compute_unit_mask": cu_mask
    }
    return task_config

def competitor_configs(cu_mask):
    """ Returns a list of competitor configs, each of which uses the given CU
    mask. """
    competitor1 = {
        "filename": "./bin/huge_kernels.so",
        "log_name": "results/competitor1.json",
        "thread_count": 256,
        "block_count": 4,
        "compute_unit_mask": cu_mask,
        "additional_info": {
            "repetitions": 500,
            "kernel_selection": "add"
        }
    }
    competitor2 = {
        "filename": "./bin/random_walk.so",
        "log_name": "results/competitor2.json",
        "thread_count": 512,
        "block_count": 128,
        "compute_unit_mask": cu_mask,
        "additional_info": {
            "iterations": 1500,
            "buffer_size": 16 * 1024 * 1024
        }
    }
    competitor3 = {
        "filename": "./bin/dwt2d.so",
        "log_name": "results/competitor3.json",
        "thread_count": 0,
        "block_count": 0,
        "compute_unit_mask": cu_mask
    }
    #return [competitor1, competitor2, competitor3]
    return [competitor2]

def top_config():
    """ Returns the top-level config, without the "plugins" field filled in.
    """
    return {
        "name": "Response-time distribution vs. CU Partition",
        "max_iterations": 1000,
        "max_time": 0,
        "gpu_device_id": 1,
        "use_processes": False,
        "pin_cpus": True,
        "do_warmup": True,
        "plugins": []
    }

def run_process(config):
    config_string = json.dumps(config)
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config_string)

print "Running isolated test"
overall_config = top_config()
task = task_config("32 CUs, no competitors",
    "results/cu_management_no_comp_32.json", "0xffffffff")
overall_config["plugins"].append(task)
run_process(overall_config)

print "Running isolated test with half CUs"
overall_config = top_config()
task = task_config("16 CUs, no competitors",
    "results/cu_management_no_comp_16.json", "0xffff0000")
overall_config["plugins"].append(task)
run_process(overall_config)

print "Running test sharing all CUs with the competitors"
overall_config = top_config()
task = task_config("32 CUs, with competitors",
    "results/cu_management_comp_32.json", "0xffffffff")
overall_config["plugins"].append(task)
for c in competitor_configs("0xffffffff"):
    overall_config["plugins"].append(c)
run_process(overall_config)

print "Running test using 16 CUs partitioned from competitors"
overall_config = top_config()
task = task_config("16 CUs, partitioned",
    "results/cu_management_partitioned_16.json", "0xffff0000")
overall_config["plugins"].append(task)
for c in competitor_configs("0x0000ffff"):
    overall_config["plugins"].append(c)
run_process(overall_config)

