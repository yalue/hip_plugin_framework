# This script is intended to generate data for a scatterplot of vector_add
# performance in the face of increasingly many competing HSA queues.
#
# NOTE: This script isn't currently exhibiting expected results (it just hangs
# with too many queues). In other words, it doesn't measure what it's supposed
# to! I'm just committing it so I can keep it for reference, when I return to
# this problem.

import json
import subprocess

def generate_config(competitor_count, stream_count):
    """ Returns a JSON string containing a config. The config pits vector_add
    against a number of dummy_streams instances, specified by competitor_count.
    The vector_add plugin always does the same amount of work, so the intention
    behind this test is to vary the number of streams it's competing against.
    Values of streams_per_competitor over 4 shouldn't matter, unless the
    GPU_MAX_HW_QUEUES environment variable is over 4. """

    queue_count = competitor_count * stream_count
    # The vector add plugin should launch exactly 10 blocks--since it has 1024
    # threads per block and 10 * 1024 elements per vector.
    measured_task_config = {
        "label": str(queue_count),
        "log_name": "results/test_oversub_vs_%d_queues.json" % (queue_count,),
        "filename": "./bin/vector_add.so",
        "thread_count": 1024,
        "block_count": 1,
        "additional_info": {
            "vector_length": 10 * 1024,
            "skip_copy": True
        }
    }

    # I think it's a bit more realistic to trigger oversubscription using a
    # large number of competitors rather than using CU masks to have a single
    # competitor with tons of streams.
    competitor_config = {
        "filename": "./bin/dummy_streams.so",
        "log_name": "/dev/null",
        "thread_count": 1,
        "block_count": 1,
        "additional_info": {
            "stream_count": stream_count,
        }
    }

    # Create our list of plugins with the measured task and competitors.
    plugin_list = [measured_task_config]
    for i in range(competitor_count):
        # We need to give each competitor a unique log name.
        plugin_list.append(competitor_config.copy())
        log_name = "results/dummy_streams_competitor_%d.json" % (i,)
        plugin_list[i + 1]["log_name"] = log_name

    name = "Vector Add Performance vs. Number of Queues"
    overall_config = {
        "name": name,
        "max_iterations": 100,
        "max_time": 0,
        "gpu_device_id": 0,
        "use_processes": True,
        "do_warmup": True,
        "omit_block_times": True,
        "sync_every_iteration": True,
        "plugins": plugin_list
    }
    return json.dumps(overall_config)

def run_process(competitor_count, stream_count):
    """ This function actually kicks off the plugin framework instance with the
    measured task and the queue-wasting competitor. """
    config = generate_config(competitor_count, stream_count)
    print("Testing with %d competitors using %d streams each." % (
        competitor_count, stream_count))
    process = subprocess.Popen(["./bin/runner", "-"], stdin=subprocess.PIPE)
    process.communicate(input=config)

if __name__ == "__main__":
    for competitor_count in range(0, 10):
        run_process(competitor_count, 3)
#    for stream_count in range(0, 32):
#        run_process(1, stream_count)

