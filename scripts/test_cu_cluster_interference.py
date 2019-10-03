# This script tests the impact of CU cluster on interference.
import json
import subprocess

def get_cu_mask(total_count, active_cu):
    """ Returns a compute-unit mask as a binary string, with a single compute
    unit set. """
    a = ["0"] * total_count
    a[active_cu] = "1"
    return "".join(a)

def generate_config(total_count, active_cu):
    """ Returns a JSON string containing a config. The active CU is the one the
    competitor workload will run on. The workload we're measuring will always
    run on CU 0. """
    # NOTE: This may take a very long time if you don't cut down on the
    # resolution of the image being generated.
    competitor_cu_mask = get_cu_mask(total_count, active_cu)
    cu_mask = get_cu_mask(total_count, 0)
    task_config = {
        "label": str(active_cu),
        "log_name": "results/active_cu%d.json" % (active_cu),
        "filename": "./bin/huge_kernels.so",
        "thread_count": 256,
        "block_count": 1,
        "compute_unit_mask": cu_mask,
        "additional_info": {
            "repetitions": 1000,
            "kernel_selection": "add",
        }
    }
    competitor_config = {
        "label": "competitor",
        "log_name": "results/cu_cluster_competitor.json",
        "filename": "./bin/huge_kernels.so",
        "thread_count": 256,
        "block_count": 1,
        "compute_unit_mask": competitor_cu_mask,
        "additional_info": {
            "repetitions": 1000,
            "kernel_selection": "add"
        }
    }
    competitor2_config = {
        "label": "competitor",
        "log_name": "results/cu_cluster_competitor2.json",
        "filename": "./bin/huge_kernels.so",
        "thread_count": 256,
        "block_count": 1,
        "compute_unit_mask": competitor_cu_mask,
        "additional_info": {
            "repetitions": 1000,
            "kernel_selection": "add"
        }
    }
    competitor3_config = {
        "label": "competitor",
        "log_name": "results/cu_cluster_competitor3.json",
        "filename": "./bin/huge_kernels.so",
        "thread_count": 256,
        "block_count": 1,
        "compute_unit_mask": competitor_cu_mask,
        "additional_info": {
            "repetitions": 1000,
            "kernel_selection": "add"
        }
    }
    overall_config = {
        "name": "CU Cluster Overlap vs. Interference",
        "max_iterations": 10,
        "max_time": 0,
        "gpu_device_id": 1,
        "pin_cpus": True,
        "sync_every_iteration": True,
        "use_processes": False,
        "do_warmup": True,
        "plugins": [
            task_config,
            competitor_config,
            competitor2_config,
            competitor3_config
        ]
    }
    return json.dumps(overall_config)

def run_process(total_cus, active_cu):
    config = generate_config(total_cus, active_cu)
    print "Starting test with competitor on CU %d." % (active_cu)
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config)

# This test was designed for the RX 570, with 32 compute units.
for i in range(31):
    run_process(32, i + 1)

