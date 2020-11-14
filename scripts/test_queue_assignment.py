# This script is intended to measure the performance of a task and an identical
# competitor, while only varying HW queue to which the competitor is assigned.
#
# In this version of the script (as opposed to test_queue_assignment_2.py),
# queue assignment is manipulated only by changing the order in which a set of
# tasks is initialized.
#
# NOTE: I have not been able to get particularly good results with this script,
# but I'm committing it simply to be a point of reference for later.
import copy
import json
import subprocess
import time

def generate_configs():
    """ Returns a list of JSON strings, each of which contains a config that
    runs the same plugins, but initializes them in a different order to
    (hopefully) change queue mappings. """
#    measured_base_config = {
#        "label": "REPLACE_ME",
#        "log_name": "results/REPLACE_ME.json",
#        "filename": "./bin/matrix_multiply.so",
#        "thread_count": 1024,
#        "block_count": 1,
#        "additional_info": {
#            "matrix_width": 1024,
#            "skip_copy": True
#        }
#    }
    measured_base_config = {
        "label": "REPLACE ME",
        "log_name": "REPLACE ME",
        "filename": "./bin/counter_spin.so",
        "thread_count": 256,
        "block_count": 1024 * 10,
        "additional_info": 1000
    }
    # We'll create a bunch of copies of the base config, first to define the
    # competitor, and then to just specify a different log name and label for
    competitor_config = copy.deepcopy(measured_base_config)
    competitor_config["log_name"] = "/dev/null"
    competitor_config["label"] = "Competitor"
    measured_1_config = copy.deepcopy(measured_base_config)
    measured_1_config["log_name"] = "results/queue_assignment_isolated.json"
    measured_1_config["label"] = "Isolated"
    measured_2_config = copy.deepcopy(measured_base_config)
    measured_2_config["log_name"] = "results/queue_assignment_0_sep.json"
    measured_2_config["label"] = "No queues created before competitor"
    measured_3_config = copy.deepcopy(measured_base_config)
    measured_3_config["log_name"] = "results/queue_assignment_1_sep.json"
    measured_3_config["label"] = "1 queue created before competitor"
    measured_4_config = copy.deepcopy(measured_base_config)
    measured_4_config["log_name"] = "results/queue_assignment_2_sep.json"
    measured_4_config["label"] = "2 queues created before competitor"
    measured_5_config = copy.deepcopy(measured_base_config)
    measured_5_config["log_name"] = "results/queue_assignment_3_sep.json"
    measured_5_config["label"] = "3 queues created before competitor"
    dummy_config = {
        "filename": "./bin/dummy_streams.so",
        "log_name": "/dev/null",
        "thread_count": 1,
        "block_count": 1,
        "compute_unit_mask": "0xdeadbeef",
        "additional_info": {
            "stream_count": 1,
            "use_cu_mask": False # May change this later...
        }
    }
    dummy1 = copy.deepcopy(dummy_config)
    dummy2 = copy.deepcopy(dummy_config)
    dummy3 = copy.deepcopy(dummy_config)

    plugin_lists = []
    plugin_lists.append([measured_1_config])
    plugin_lists.append([measured_2_config, competitor_config, dummy1, dummy2, dummy3])
    plugin_lists.append([measured_3_config, dummy1, competitor_config, dummy2, dummy3])
    plugin_lists.append([measured_4_config, dummy1, dummy2, competitor_config, dummy3])
    plugin_lists.append([measured_5_config, dummy1, dummy2, dummy3, competitor_config])

    # Assign each member of each list an initialization delay time to ideally
    # force them to start in the order they appear in the list (verify this
    # manually by watching the order printed to stdout by the framework).
    for i in range(len(plugin_lists)):
        # We need to jump through hoops here to change the delays on a copy--
        # otherwise we'll end up making all of the delays the same as the last
        # list.
        list_copy = copy.deepcopy(plugin_lists[i])
        delay_seconds = 0.0
        for j in range(len(list_copy)):
            list_copy[j]["initialization_delay"] = delay_seconds
            delay_seconds += 2.0
        plugin_lists[i] = list_copy

    # The top-level config JSON object. We'll just swap out the plugins list to
    # generate each config. Setting initialize_in_order to True is *essential*
    # for this experiment.
    overall_config = {
        "name": "Queue assignment vs. Performance",
        "max_iterations": 100000,
        "max_time": 10,
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
        print("Done with test. Waiting a bit.\n\n")
        time.sleep(5.0)

