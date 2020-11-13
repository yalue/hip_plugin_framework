# This script parses all JSON results files, and prints various stats about
# data contained in the files.
import argparse
import glob
import json
import numpy
import sys

def get_counts(content):
    """ Returns a tuple (iteration_count, kernel_count) from the times
    contained in the file. """
    iteration_count = 0
    kernel_count = 0
    for t in content["times"]:
        if "cpu_times" in t:
            iteration_count += 1
        if "kernel_name" in t:
            kernel_count += 1
    return (iteration_count, kernel_count)

def compute_stats(data):
    """ Returns the min, max, med, mean, and stddev (in that order) of the
    given data. """
    data.sort()
    med = data[int(len(data) / 2)]
    mean = numpy.mean(data)
    std = numpy.std(data)
    return data[0], data[-1], med, mean, std


def print_distribution_stats(label, times, indent_prefix="  "):
    """ Takes a label string, a distribution of times, and a prefix for
    indentation.  Computes stats for the distribution and formats the text. """
    if len(times) == 0:
        print("%s%s times: No data available" % (indent_prefix, label))
        return
    minimum, maximum, median, mean, stddev = compute_stats(times)
    msg = "%s%s times (%d samples): Min = %f, Max = %f, Median = %f, Mean = %f"
    msg += ", Standard deviation = %f"
    msg = msg % (indent_prefix, label, len(times), minimum, maximum, median,
        mean, stddev)
    print(msg)

def get_times_by_key(content, key):
    to_return = []
    for t in content["times"]:
        if key not in t:
            continue
        times = t[key]
        to_return.append(times[1] - times[0])
    return to_return

def get_copy_in_times(content):
    return get_times_by_key(content, "copy_in_times")

def get_copy_out_times(content):
    return get_times_by_key(content, "copy_out_times")

def get_execute_times(content):
    return get_times_by_key(content, "execute_times")

def get_iteration_times(content):
    return get_times_by_key(content, "cpu_times")

def get_unique_kernel_names(content):
    kernel_names = {}
    for t in content["times"]:
        if "kernel_name" not in t:
            continue
        name = t["kernel_name"]
        kernel_names[name] = True
    to_return = []
    for k in kernel_names:
        to_return.append(k)
    return to_return

def get_kernel_launch_times(content, kernel_name):
    to_return = []
    for t in content["times"]:
        if "kernel_name" not in t:
            continue
        if t["kernel_name"] != kernel_name:
            continue
        if len(t["kernel_launch_times"]) != 3:
            continue
        # times[2] contains the time after hipDeviceSynchronize returns.
        times = t["kernel_launch_times"]
        to_return.append(times[2] - times[0])
    return to_return

def get_block_times(content, kernel_name):
    to_return = []
    for t in content["times"]:
        if "kernel_name" not in t:
            continue
        if t["kernel_name"] != kernel_name:
            continue
        block_times = t["block_times"]
        # Read block start and end times from the (potentially long) list of
        # block times until it's empty.
        while len(block_times) != 0:
            block_start = block_times.pop(0)
            block_end = block_times.pop(0)
            to_return.append(block_end - block_start)
    return to_return

def print_single_file_stats(filename, content):
    """ Takes a file's name, and its parsed JSON content. Computes basic stats
    about the file and prints them to stdout. """
    print("File %s:" % (filename,))
    print("  Plugin: %s" % (content["plugin_name"],))
    print("  Scenario: %s" % (content["scenario_name"],))
    if "label" in content:
        print("  Label: %s" % (content["label"],))
    iterations, kernels = get_counts(content)
    print("  Ran %d iterations, launching %d kernels" % (iterations, kernels))

    # We'll group all CPU-based measurements together.
    print_distribution_stats("Copy in", get_copy_in_times(content))
    print_distribution_stats("Execute", get_execute_times(content))
    print_distribution_stats("Copy out", get_copy_out_times(content))
    print_distribution_stats("Full iteration", get_iteration_times(content))

    # We'll group kernel times by kernel.
    kernel_names = get_unique_kernel_names(content)
    for kernel_name in kernel_names:
        print("  Times for kernel %s:" % (kernel_name,))
        launch_times = get_kernel_launch_times(content, kernel_name)
        block_times = get_block_times(content, kernel_name)
        print_distribution_stats("Kernel launch", launch_times, "    ")
        print_distribution_stats("Thread block", block_times, "    ")

def print_stats(filenames):
    """ Takes a list of filenames, and wraps calls to print_single_file_stats.
    """
    for name in filenames:
        parsed = None
        with open(name) as f:
            parsed = json.loads(f.read())
            if len(parsed["times"]) < 2:
                print("%s: No recorded times in file." % (name))
                continue
        print_single_file_stats(name, parsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory",
        help="Directory containing result JSON files.", default='./results')
    args = parser.parse_args()
    filenames = glob.glob(args.directory + "/*.json")
    print_stats(filenames)

