# This script is intended to measure the performance of a task and an identical
# competitor, while only varying HW queue to which the competitor is assigned.
#
# NOTE: I have not been getting particularly good results with this script--
# I am committing it only to serve as reference for later. If it's not changed
# significantly I will probably delete it.
import copy
import json
import subprocess
import time

def generate_configs():
    """ Returns a list of JSON strings, each of which contains a config that
    runs the same plugins, but initializes them in a different order to
    (hopefully) change queue mappings. """
    measured_base_config = {
        "label": "REPLACE_ME",
        "log_name": "results/REPLACE_ME.json",
        "filename": "./bin/matrix_multiply.so",
        "initialization_delay": 0.0,
        "thread_count": [32, 32],
        "block_count": 1,
        "additional_info": {
            "matrix_width": 1024,
            "skip_copy": True
        }
    }
    # We'll create several copies of the base config, first to define the
    # identical competitor, and then just to specify the different log names.
    competitor_config = copy.deepcopy(measured_base_config)
    competitor_config["initialization_delay"] = 8.0
    competitor_config["log_name"] = "/dev/null"
    competitor_config["label"] = "Competitor"
    measured_1_config = copy.deepcopy(measured_base_config)
    measured_1_config["log_name"] = "results/queue_assignment_0_sep.json"
    measured_1_config["label"] = "No queues created before competitor"
    measured_2_config = copy.deepcopy(measured_base_config)
    measured_2_config["log_name"] = "results/queue_assignment_1_sep.json"
    measured_2_config["label"] = "1 queue created before competitor"
    measured_3_config = copy.deepcopy(measured_base_config)
    measured_3_config["log_name"] = "results/queue_assignment_2_sep.json"
    measured_3_config["label"] = "2 queues created before competitor"
    measured_4_config = copy.deepcopy(measured_base_config)
    measured_4_config["log_name"] = "results/queue_assignment_3_sep.json"
    measured_4_config["label"] = "3 queues created before competitor"

    # This dummy config will be copied to vary the number of streams to create.
    dummy_config = {
        "filename": "./bin/dummy_streams.so",
        "log_name": "/dev/null",
        "thread_count": 1,
        "block_count": 1,
        "compute_unit_mask": "0xdeadbeef",
        "initialization_delay": 4.0,
        "additional_info": {
            "stream_count": 0,
            "use_cu_mask": True
        }
    }
    dummy1 = copy.deepcopy(dummy_config)
    dummy1["additional_info"]["stream_count"] = 1
    dummy2 = copy.deepcopy(dummy_config)
    dummy2["additional_info"]["stream_count"] = 2
    dummy3 = copy.deepcopy(dummy_config)
    dummy3["additional_info"]["stream_count"] = 3

    # Note how we used initialization_delay to always try to force the
    # initialization order: measured -> dummy -> competitor.
    plugin_lists = []
    plugin_lists.append([measured_1_config, competitor_config])
    plugin_lists.append([measured_2_config, dummy1, competitor_config])
    plugin_lists.append([measured_3_config, dummy2, competitor_config])
    plugin_lists.append([measured_4_config, dummy2, competitor_config])


    # The top-level config JSON object. We'll just swap out the plugins list to
    # generate each config. Setting initialize_in_order to True is *essential*
    # for this experiment.
    overall_config = {
        "name": "Queue assignment vs. Performance",
        "max_iterations": 1000000,
        "max_time": 10.0,
        "gpu_device_id": 0,
        "use_processes": True,
        "do_warmup": True,
        "omit_block_times": True,
        "sync_every_iteration": False,
        "plugins": []
    }

    to_return = []
    for l in plugin_lists:
        overall_config["plugins"] = l
        to_return.append(json.dumps(overall_config))
    return to_return

if __name__ == "__main__":
    configs = generate_configs()
    for c in configs:
        p = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
        p.communicate(input=c)
        # This seems to help with hangs during load.
        print("Done. Waiting a bit before the next test.")
        time.sleep(4.0)

