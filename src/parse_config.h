// This file defines the interface and types needed when parsing the JSON
// configuration files used by runner.c.
#ifndef PARSE_CONFIG_H
#define PARSE_CONFIG_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include "plugin_interface.h"

#define USE_DEFAULT_CPU_CORE (-1)

// Holds the configuration for a single plugin, as specified by the JSON format
// given in the README.
typedef struct {
  // The name of the plugin's .so file
  char *filename;
  // The name of the plugin's JSON log, relative to base_result_directory.
  char *log_name;
  // An extra label or name for the plugin, included in its JSON log file.
  char *label;
  // The number of threads to request the plugin to use
  int thread_count;
  // The number of blocks to request the plugin to use
  int block_count;
  // The size, in bytes, of the input data the plugin should generate or use
  uint64_t data_size;
  // A string containing an additional user-defined argument to pass to the
  // plugin during initialization. May be either NULL or empty if unspecified.
  char *additional_info;
  // A bit mask where 1 bits enable the respective compute unit and 0 disables
  // them. Defaults to all 1s.
  uint64_t compute_unit_mask[COMPUTE_UNIT_MASK_ENTRIES];
  // The maximum number of iterations for this plugin alone; will override the
  // global limit if set (0 = unlimited, negative = unset).
  int64_t max_iterations;
  // The maximum number of seconds to run this plugin alone; will override the
  // global limit if set (0 = unlimited, negative = unset).
  double max_time;
  // The number of seconds for which the plugin should sleep before starting.
  // If 0 or negative, it won't sleep at all.
  double release_time;
  // The CPU core to pin this plugin to. Ignored if negative.
  int cpu_core;
} PluginConfiguration;

// Holds default settings for all plugins, and a list of individual plugins
// with their specific settings.
typedef struct {
  // The default cap on the number of iterations each plugin will run.
  // Unlimited if 0 or lower.
  int64_t max_iterations;
  // The default cap on the number of seconds each plugin will run. Unlimited
  // if 0 or lower.
  double max_time;
  // Set to 0 if each plugin should be run in a separate thread. Set to 1 if
  // each should be run in a child process instead.
  int use_processes;
  // The device ID for the GPU to run plugins on.
  int gpu_device_id;
  // The path to the base directory in which plugin's log files are stored.
  char *base_result_directory;
  // The name of the scenario being tested.
  char *scenario_name;
  // If zero, CPU assignment is either handled by the system or taken from each
  // plugin's cpu_core setting. If nonzero, plugins are distributed evenly
  // accross CPU cores.
  int pin_cpus;
  // If zero, iterations of individual plugins run as soon as previous
  // iterations complete. If 1, then every plugin starts each iteration only
  // after the previous iteration of every plugin has completed.
  int sync_every_iteration;
  // The number of entries in the plugins list. Must never be 0.
  int plugin_count;
  // The list of plugins to run.
  PluginConfiguration *plugins;
} GlobalConfiguration;

// Parses a JSON configuration string, and allocates and returns a
// GlobalConfiguration struct. Returns NULL on error. When no longer needed,
// the returned pointer must be passed to FreeGlobalConfiguration. May print to
// stdout on error.
GlobalConfiguration* ParseConfiguration(const char *content);

void FreeGlobalConfiguration(GlobalConfiguration *config);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // PARSE_CONFIG_H
