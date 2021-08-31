import glob
import itertools
import json
import matplotlib.pyplot as plot
import numpy
import re
import sys
import argparse
from scipy import stats

def get_total_kernel_times(plugin):
    """ This does extra processing on the plugin's results in order to get
    "total kernel time", which doesn't correspond to a real JSON key. Instead,
    it consists of the sum of actual kernel times, from launch to
    after the synchronization, for each iteration of the plugin. """
    to_return = []
    current_total = 0.0
    for t in plugin["times"]:
        # Reset the running total whenever we've encountered a new iteration,
        # as indicated by a JSON record containing CPU times.
        if "cpu_times" in t:
            # The check against 0.0 takes care of both the first iteration, and
            # better handles benchmarks without kernel times.
            if current_total != 0.0:
                to_return.append(current_total)
            current_total = 0.0
            continue
        if "kernel_launch_times" not in t:
            continue
        start = t["kernel_launch_times"][0]
        end = t["kernel_launch_times"][-1]
        current_total += end - start

    return to_return

def get_plugin_raw_values(plugin, times_key):
    """Takes a parsed plugin result JSON file and returns raw values of the
    selected times for the plugin.  The times_key is used to specify which
    specific time measurement (from objects in the times array) to use."""
    raw_values = []
    # Do some special handling to compute the total_kernel_times, since it
    # isn't actually a key in the JSON.
    if times_key == "total_kernel_times":
        t = get_total_kernel_times(plugin)
        for i in range(len(t)):
            t[i] *= 1000.0
        return t
    for t in plugin["times"]:
        if not times_key in t:
            continue
        times = t[times_key]
        for i in range(len(times) // 2):
            start_index = i * 2
            end_index = i * 2 + 1
            milliseconds = (times[end_index] - times[start_index]) * 1000.0
            raw_values.append(milliseconds)
    return raw_values

def nice_sort_key(label):
    """If a label contains numbers, this will prevent sorting them
    lexicographically."""
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    return [tryint(c) for c in re.split(r'([0-9]+)', label)]

def plugin_sort_key(plugin):
    """Returns the key that may be used to sort plugin results by label."""
    if not "label" in plugin:
        return ""
    return nice_sort_key(plugin["label"])

all_styles = None
def get_line_styles():
    """Returns a list of line style possibilities, that includes more options
    than matplotlib's default set that includes only a few solid colors."""
    global all_styles
    if all_styles is not None:
        return all_styles
    color_options = [
        "blue",
        "green",
        "red",
        "cyan",
        "magenta",
        "y",
        "black"
    ]
    style_options = [
        "-",
        "--",
        "-.",
        ":"
    ]
    marker_options = [
        None,
        "o",
        "v",
        "s",
        "*",
        "+",
        "D"
    ]
    # Build a combined list containing every style combination.
    all_styles = []
    for m in marker_options:
        for s in style_options:
            to_add = {}
            if m is not None:
                to_add["marker"] = m
                to_add["markevery"] = 0.1
            to_add["ls"] = s
            all_styles.append(to_add)
    return all_styles

def add_plot_padding(axes):
    """Takes matplotlib axes, and adds some padding so that lines close to
    edges aren't obscured by tickmarks or the plot border."""
    y_limits = axes.get_ylim()
    y_range = y_limits[1] - y_limits[0]
    y_pad = y_range * 0.05
    x_limits = axes.get_xlim()
    x_range = x_limits[1] - x_limits[0]
    x_pad = x_range * 0.05
    axes.set_ylim(y_limits[0] - y_pad, y_limits[1] + y_pad)
    axes.set_xlim(x_limits[0] - x_pad, x_limits[1] + x_pad)
    axes.xaxis.set_ticks(numpy.arange(x_limits[0], x_limits[1] + x_pad,
        x_range / 5.0))
    axes.yaxis.set_ticks(numpy.arange(y_limits[0], y_limits[1] + y_pad,
        y_range / 5.0))
    return

def get_x_range(raw_data, sample_count = 2000):
    """Takes a list of raw data, and returns a range of points to use as the
    x-values of the KDE plot."""
    x_min = min(raw_data)
    x_max = max(raw_data)
    return numpy.arange(x_min, x_max, (x_max - x_min) / 2000.0)

def plot_scenario(plugins, name, times_key):
    """Takes a list of parsed benchmark results and a scenario name and
    generates a PDF plot of CPU times for the scenario. See
    get_plugin_raw_values for an explanation of the times_key argument."""
    plugins = sorted(plugins, key = plugin_sort_key)
    style_cycler = itertools.cycle(get_line_styles())
    raw_data_array = []
    labels = []
    c = 0
    for p in plugins:
        c += 1
        label = "%d: %s" % (c, p["plugin_name"])
        if "label" in p:
            label = p["label"]
        labels.append(label)
        raw_data = get_plugin_raw_values(p, times_key)
        raw_data_array.append(raw_data)
    figure = plot.figure()
    figure.suptitle(name + ": " + times_key)
    axes = figure.add_subplot(1, 1, 1)
    # Make the axes track data exactly, we'll manually add padding later.
    axes.autoscale(enable=True, axis='both', tight=True)
    for i in range(len(raw_data_array)):
        density = stats.kde.gaussian_kde(raw_data_array[i])
        x = get_x_range(raw_data_array[i])
        axes.plot(x, density(x), label=labels[i], **(next(style_cycler)))
    add_plot_padding(axes)
    axes.set_xlabel("Time (milliseconds)")
    axes.set_ylabel("Density")
    legend = plot.legend()
    legend.set_draggable(True)
    return figure

def show_plots(filenames, times_key):
    """Takes a list of filenames, and generates one plot per scenario found in
    the files. See get_plugin_raw_values for an explanation of the times_key
    argument."""
    parsed_files = []
    for name in filenames:
        with open(name) as f:
            parsed_files.append(json.loads(f.read()))
    # Group the files by scenario
    scenarios = {}
    for plugin_result in parsed_files:
        scenario = plugin_result["scenario_name"]
        if not scenario in scenarios:
            scenarios[scenario] = []
        scenarios[scenario].append(plugin_result)
    figures = []
    for scenario in scenarios:
        print(scenario)
        figures.append(plot_scenario(scenarios[scenario], scenario, times_key))
    plot.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory",
        help="Directory containing result JSON files.", default='./results')
    parser.add_argument("-k", "--times_key",
        help="JSON key name for the time property to be plot.",
        default="cpu_times")
    parser.add_argument("-r", "--regex",
        help="Regex for JSON files to be processed",
        default="*.json")
    args = parser.parse_args()
    filenames = glob.glob(args.directory + "/" + args.regex)
    show_plots(filenames, args.times_key)

