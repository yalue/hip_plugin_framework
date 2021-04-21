# This script is intended to generate some CDF plots showing matrix-multiply
# performance when the kernel is preempted a varying number of times.
import json
import subprocess

def generate_config(preempt_count):
    """ Returns a JSON config pitting matrix_multiply against dummy_lock_gpu,
    where dummy_gpu_lock's preempt_count is varied. """
    dlg_config = {
        "filename": "./bin/dummy_lock_gpu.so",
        "log_name": "/dev/null",
        "thread_count": 1,
        "block_count": 1,
        "additional_info": {
            "preempt_count": preempt_count
        }
    }
    mm_config = {
        "filename": "./bin/matrix_multiply.so",
        "log_name": "results/preempted_%d_mm.json" % (preempt_count,),
        "label": "Matrix Mult., Preempted %d times" % (preempt_count,),
        "job_deadline": 0.060,
        "thread_count": [32, 32],
        "block_count": 1,
        "additional_info": {
            "matrix_width": 1024,
            "skip_copy": True
        }
    }
    overall_config = {
        "name": "Preemption-overhead test",
        "max_iterations": 100,
        "max_time": 0,
        "gpu_device_id": 0,
        "use_processes": True,
        "omit_block_times": True,
        "do_warmup": True,
        "sync_every_iteration": True,
        #"pin_cpus": True,
        "plugins": [dlg_config, mm_config]
    }
    return json.dumps(overall_config)

def run_process(preempt_count):
    """ Starts the process that will run the tests with the given preempt
    count. """
    config = generate_config(preempt_count)
    config = bytes(config, "utf-8")
    print("Starting test with preempt_count = %d" % (preempt_count,))
    proccess = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    proccess.communicate(input=config)

if __name__ == "__main__":
    for i in range(0, 4):
        run_process(i)

