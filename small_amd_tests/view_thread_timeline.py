# This script generates and displays a stack plot showing the timeline of
# active threads and blocks from thread_results.json (generated using
# test_thread_scheduling.cpp).
import json
import matplotlib.pyplot as plot
import numpy

def adjust_time_offsets(f):
    """ Takes the input data and adjusts all of the times to make the earliest
    time 0 and the rest of the times relative to that. """
    earliest = 1e15
    for b in f["blocks"]:
        for t in b["start_times"]:
            if t < earliest:
                earliest = t
    # The length of the start times must be equal to the length of end times.
    for b in f["blocks"]:
        for i in range(len(b["start_times"])):
            b["start_times"][i] -= earliest
            b["end_times"][i] -= earliest

def blocks_start_end(all_blocks):
    """ Returns a single "block" dict containing start and end times, except
    it contains the start and end times of entire blocks. """
    to_return = {}
    to_return["start_times"] = []
    to_return["end_times"] = []
    for b in all_blocks:
        first_start = 1e15
        last_end = 0
        for t in b["start_times"]:
            if t < first_start:
                first_start = t
        for t in b["end_times"]:
            if t > last_end:
                last_end = t
        to_return["start_times"].append(first_start)
        to_return["end_times"].append(last_end)
    return to_return

def get_block_timeline(b):
    """ Takes a "block" dict, and returns two arrays: a list of times, and
    a list of the number of active threads at each corresponding time. Times
    are in thousands of clock cycles. """
    start_times = b["start_times"]
    end_times = b["end_times"]
    # Sort in reverse order so that the earliest elements are at the ends of
    # the array and can be popped first.
    start_times.sort(reverse = True)
    end_times.sort(reverse = True)
    timeline_times = []
    timeline_values = []
    timeline_times.append(0.0)
    timeline_values.append(0)
    current_thread_count = 0
    while True:
        if (len(start_times) == 0) and (len(end_times) == 0):
            break
        if len(end_times) == 0:
            print("Error! The last block end time was before a start time.")
            exit(1)
        current_time = 0.0
        previous_thread_count = current_thread_count
        if len(start_times) != 0:
            # Get the next closest time, be it a start or an end time. The <=
            # is important, since we always want increment the block count
            # before decrementing in the case when a block starts at the same
            # time another ends.
            if start_times[-1] == end_times[-1]:
                # A thread started and ended at the same time, don't change the
                # current thread count.
                current_time = start_times.pop()
                end_times.pop()
            elif start_times[-1] <= end_times[-1]:
                # A thread started, so increase the thread count
                current_time = start_times.pop()
                current_thread_count += 1
            else:
                # A thread ended, so decrease the thread count
                current_time = end_times.pop()
                current_thread_count -= 1
        else:
            # Only end times are left, so keep decrementing the thread count.
            current_time = end_times.pop()
            current_thread_count -= 1
        # Make sure that changes between numbers of running threads are abrupt.
        # Do this by only changing the number of blocks at the instant they
        # actually change rather than interpolating between two values.
        timeline_times.append(current_time)
        timeline_values.append(previous_thread_count)
        # Finally, append the new thread count.
        timeline_times.append(current_time)
        timeline_values.append(current_thread_count)
    return [timeline_times, timeline_values]

def rebase_timeline(original_times, original_values, new_times):
    """ Returns a timeline like get_block_timeline, but ensures that the times
    array contains all of the times in the new_times array. """
    timeline_times = []
    timeline_values = []
    original_times.reverse()
    original_values.reverse()
    # Clone the new_times so we don't trash the input.
    new_times = list(new_times)
    new_times.reverse()

    # Note: this assumes that the original timeline will have at most two
    # duplicate times in a row, since the new_times array will have two copies
    # of every time.
    while len(new_times) > 0:
        current_time = new_times.pop()
        current_value = original_values[-1]
        timeline_times.append(current_time)
        timeline_values.append(current_value)
        if current_time == original_times[-1]:
            original_times.pop()
            original_values.pop()
    return [timeline_times, timeline_values]

def make_common_x_axis(block_timelines):
    """ Takes a list of block timelines and modifies it so that all of the
    timelines have identical values on their X axes. """
    all_times = []
    for b in block_timelines:
        times = b[0]
        for t in times:
            all_times.append(t)
    all_times = sorted(set(all_times))
    # all_times now contains a sorted superset of all timelines' "time" arrays
    # However, we still need to duplicate every entry so that individual time
    # changes can appear abrupt rather than being interpolated.
    tmp = []
    for t in all_times:
        tmp.append(t)
        tmp.append(t)
    all_times = tmp

    for i in range(len(block_timelines)):
        old_times = block_timelines[i][0]
        old_values = block_timelines[i][1]
        rebased = rebase_timeline(old_times, old_values, all_times)
        block_timelines[i] = rebased

def get_stackplot_values(timelines):
    # Track indices into the list of times and values from each plugin as
    # we build an aggregate list.
    times_lists = []
    values_lists = []
    indices = []
    new_times = []
    new_values = []

    for t in timelines:
        times_lists.append(t[0])
        values_lists.append(t[1])
        indices.append(0)
        new_values.append([])

    # Selects the next smallest time we need to add to our output list.
    def current_min_time():
        current_min = 1e99
        for i in range(len(indices)):
            n = indices[i]
            if n >= len(times_lists[i]):
                continue
            if times_lists[i][n] < current_min:
                current_min = times_lists[i][n]
        return current_min

    # Returns true if we've seen all the times in every list.
    def all_times_done():
        for i in range(len(indices)):
            if indices[i] < len(times_lists[i]):
                return False
        return True

    # Moves to the next time for each input list if the current time is at the
    # head of the list.
    def update_indices(current_time):
        for i in range(len(indices)):
            n = indices[i]
            if n >= len(times_lists[i]):
                continue
            if times_lists[i][n] == current_time:
                indices[i] += 1

    def update_values():
        for i in range(len(indices)):
            n = indices[i]
            if n >= len(values_lists[i]):
                new_values[i].append(0)
                continue
            new_values[i].append(values_lists[i][n])

    while not all_times_done():
        current_time = current_min_time()
        new_times.append(current_time)
        update_values()
        update_indices(current_time)

    to_return = []
    to_return.append(new_times)
    for v in new_values:
        to_return.append(v)
    return to_return
input_data = None
with open("thread_results.json") as f:
    input_data = json.loads(f.read())
adjust_time_offsets(input_data)
whole_block_timeline = get_block_timeline(blocks_start_end(input_data["blocks"]))
block_timelines = []
for b in input_data["blocks"]:
    block_timelines.append(get_block_timeline(b))
v = get_stackplot_values(block_timelines)
#make_common_x_axis(block_timelines)

## At this point, the x axis should be the same for every block timeline, so
## just take the first one.
#x_values = block_timelines[0][0]
## We want an array containing every timeline's values to pass to stackplot(..)
#y_values = []
#for t in block_timelines:
#    y_values.append(t[1])

figure = plot.figure()
figure.suptitle("Threads over time")
axes = figure.add_subplot(1, 1, 1)
axes.stackplot(*v)
axes.set_ylabel("# threads")
axes.set_xlabel("Time (GPU clock)")
axes_2 = axes.twinx()
axes_2.plot(whole_block_timeline[0], whole_block_timeline[1], color='k', lw=2)
axes_2.set_ylabel("# blocks")

figure.tight_layout()
plot.savefig("/home/otternes/Desktop/threads_and_blocks.png", dpi=1200, pad_inches = 0.2)
plot.show()

