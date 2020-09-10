// This file implements the functions for parsing JSON configuration files
// defined in parse_config.h and README.md.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "third_party/cJSON.h"
#include "parse_config.h"
#include "plugin_interface.h"

// Config files are read in chunks containing this many bytes.
#define FILE_CHUNK_SIZE (4096)

// Set to 1 to use processes rather than threads by default. The readme says
// to use threads by default, so leave this as 0.
#define DEFAULT_USE_PROCESSES (0)

#define DEFAULT_BASE_RESULT_DIRECTORY "./results"

// Returns 0 if any key in the given cJSON config isn't in the given list of
// valid keys, and nonzero otherwise. The cJSON object must refer to the first
// sibling.
static int VerifyConfigKeys(cJSON *config, char **valid_keys,
  int valid_keys_count) {
  int i, found;
  while (config != NULL) {
    found = 0;
    if (!config->string) {
      printf("Found a setting without a name in the config.\n");
      return 0;
    }
    for (i = 0; i < valid_keys_count; i++) {
      if (strncmp(config->string, valid_keys[i], strlen(valid_keys[i])) == 0) {
        found = 1;
        break;
      }
    }
    if (!found) {
      printf("Unknown setting in config: %s\n", config->string);
      return 0;
    }
    config = config->next;
  }
  return 1;
}

// Ensures that all JSON key names in global config are known. Returns 0 if an
// unknown setting is found, and nonzero otherwise.
static int VerifyGlobalConfigKeys(cJSON *main_config) {
  int keys_count = 0;
  char *valid_keys[] = {
    "name",
    "max_iterations",
    "max_time",
    "use_processes",
    "gpu_device_id",
    "base_result_directory",
    "pin_cpus",
    "plugins",
    "sync_every_iteration",
    "do_warmup",
    "omit_block_times",
    "comment",
  };
  keys_count = sizeof(valid_keys) / sizeof(char*);
  return VerifyConfigKeys(main_config, valid_keys, keys_count);
}

// Ensures that all JSON key names in a plugin config are known. Returns 0 if
// an unknown setting is found, and nonzero otherwise.
static int VerifyPluginConfigKeys(cJSON *plugin_config) {
  int keys_count = 0;
  char *valid_keys[] = {
    "filename",
    "log_name",
    "label",
    "thread_count",
    "block_count",
    "additional_info",
    "max_iterations",
    "max_time",
    "release_time",
    "cpu_core",
    "compute_unit_mask",
    "comment",
  };
  keys_count = sizeof(valid_keys) / sizeof(char*);
  return VerifyConfigKeys(plugin_config, valid_keys, keys_count);
}

// Returns 1 if the given cJSON object is a boolean, and 0 otherwise.
static int IsCJSONBoolean(cJSON *o) {
  return (o->type == cJSON_True) || (o->type == cJSON_False);
}

// Converts the hexadecimal character c to its 4-bit numerical equivalent. Case
// insensitive. Returns a value over 0xf if the provided character isn't a
// valid hex value.
static uint64_t HexTo64(char c) {
  if (c < '0') return 100;
  if (c <= '9') return c - '0';
  if (c < 'A') return 100;
  if (c <= 'F') return 10 + (c - 'A');
  if (c < 'a') return 100;
  if (c <= 'f') return 10 + (c - 'a');
  return 100;
}

// Takes a hexadecimal string and interprets it as a set of bits corresponding
// to a compute unit mask. Returns 0 on error.
static int ParseComputeUnitMaskHex(const char *s, uint32_t *mask_values) {
  uint64_t tmp;
  int length, i;
  length = strlen(s);
  // Each character specifies 4 bits, and each 32-bit "entry" here holds 32
  // bits.
  if ((length * 4) > (COMPUTE_UNIT_MASK_ENTRIES * 32)) {
    printf("%s is too long to be a hexadecimal compute unit mask.\n", s);
    return 0;
  }
  // Read the string in reverse order so we can start filling in the lower bits
  // of the mask first.
  for (i = length - 1; i >= 0; i--) {
    tmp = HexTo64(s[i]);
    if (tmp > 0xf) {
      printf("Compute unit mask %s contains invalid hexadecimal characters.\n",
        s);
      return 0;
    }
    // Remember that the mask values default to all being set, so we'll need to
    // do a bit of extra bitwise stuff to unset bits that are supposed to be 0.
    mask_values[i / 8] ^= (~tmp & 0xf) << ((i % 8) * 4);
  }
  return 1;
}

// Takes a binary string and interprets it as a set of bits corresponding to a
// compute unit mask. Returns 0 on error.
static int ParseComputeUnitMaskBinary(const char *s, uint32_t *mask_values) {
  char tmp;
  int length, i;
  length = strlen(s);
  // We already checked that the length of s is short enough, so we'll just
  // start reading it in reverse order.
  for (i = length - 1; i >= 0; i--) {
    tmp = s[i];
    if (tmp == '1') {
      continue;
    } else if (tmp == '0') {
      // As before, remember that our goal is to unset the bits that should be
      // 0, since all of the mask bits default to 1.
      mask_values[i / 32] ^= 1ul << (i % 32);
    } else {
      printf("Compute unit mask %s contains invalid binary characters.\n", s);
      return 0;
    }
  }
  return 1;
}

// Attempts to parse the given string as a binary or hexadecimal number
// representing a compute unit mask. If the string starts with "0x" it will be
// treated as hexadecimal, otherwise it will be treated as binary. Returns 0 on
// error.
static int ParseComputeUnitMaskString(const char *s, uint32_t *mask_values) {
  int max_mask_length = COMPUTE_UNIT_MASK_ENTRIES * 32;
  int string_length = strnlen(s, max_mask_length + 1);
  if (string_length > max_mask_length) {
    printf("The compute unit mask string is too long to be a %d-bit mask.\n",
      COMPUTE_UNIT_MASK_ENTRIES * 32);
    return 0;
  }
  if ((string_length > 2) && (s[0] == '0') && (s[1] == 'x')) {
    return ParseComputeUnitMaskHex(s + 2, mask_values);
  }
  return ParseComputeUnitMaskBinary(s, mask_values);
}

// Takes plugin config JSON object and looks for a "compute_unit_mask" list of
// booleans. If such a list exists, this function will set the bits in the
// mask_values buffer (note that mask_values must contain the number of 64-bit
// values specified by COMPUTE_UNIT_MASK_ENTRIES. Returns 0 on error. If the
// config doesn't specify a compute unit mask for this plugin, this will simply
// set all bits in mask_values to 1.
static int ParseComputeUnitMask(cJSON *plugin_config, uint32_t *mask_values) {
  int i, j;
  cJSON *list = cJSON_GetObjectItem(plugin_config, "compute_unit_mask");
  cJSON *array_entry = NULL;
  memset(mask_values, 0xff, COMPUTE_UNIT_MASK_ENTRIES * sizeof(uint32_t));
  if (!list) return 1;
  if (list->type == cJSON_String) {
    return ParseComputeUnitMaskString(list->valuestring, mask_values);
  }
  if (list->type != cJSON_Array) {
    printf("Expected an array or a string for compute_unit_mask.");
    return 0;
  }
  array_entry = list->child;
  for (i = 0; i < COMPUTE_UNIT_MASK_ENTRIES; i++) {
    for (j = 0; j < 32; j++) {
      if (!array_entry) break;
      if (!IsCJSONBoolean(array_entry)) {
        printf("The compute_unit_mask array must contain only booleans.");
        return 0;
      }
      // The bits default to all being set, so we'll unset bits corresponding
      // to false entries in the array.
      if (array_entry->type == cJSON_False) {
        mask_values[i] ^= 1ul << j;
      }
      array_entry = array_entry->next;
    }
  }
  if (array_entry != NULL) {
    printf("The compute_unit_mask array contained more entries than supported "
      "compute units by the plugin framework. Max supported CUs: %d.\n", 64 *
      COMPUTE_UNIT_MASK_ENTRIES);
    return 0;
  }
  return 1;
}

// Parses the list of individaul plugin settings, starting with the entry given
// by list_start. The list_start entry must have already been valideated when
// this is called. On error, this will return 0 and leave the config object
// unmodified. On success, this returns 1.
static int ParsePluginList(GlobalConfiguration *config, cJSON *list_start) {
  cJSON *current_plugin = NULL;
  cJSON *entry = NULL;
  int plugin_count = 1;
  int i;
  size_t plugins_size = 0;
  PluginConfiguration *plugins = NULL;
  // Start by counting the number of plugins in the array and allocating memory
  entry = list_start;
  while (entry->next) {
    plugin_count++;
    entry = entry->next;
  }
  plugins_size = plugin_count * sizeof(PluginConfiguration);
  plugins = (PluginConfiguration *) malloc(plugins_size);
  if (!plugins) {
    printf("Failed allocating space for the plugin list.\n");
    return 0;
  }
  memset(plugins, 0, plugins_size);
  // Next, traverse the array and fill in our parsed copy.
  current_plugin = list_start;
  for (i = 0; i < plugin_count; i++) {
    if (!VerifyPluginConfigKeys(current_plugin->child)) {
      goto ErrorCleanup;
    }
    entry = cJSON_GetObjectItem(current_plugin, "filename");
    if (!entry || (entry->type != cJSON_String)) {
      printf("Missing/invalid plugin filename in the config file.\n");
      goto ErrorCleanup;
    }
    plugins[i].filename = strdup(entry->valuestring);
    if (!plugins[i].filename) {
      printf("Failed copying plugin filename.\n");
      goto ErrorCleanup;
    }
    entry = cJSON_GetObjectItem(current_plugin, "log_name");
    if (!entry || (entry->type != cJSON_String)) {
      printf("Invalid or missing plugin log_name in the config file.\n");
      goto ErrorCleanup;
    }
    plugins[i].log_name = strdup(entry->valuestring);
    if (!plugins[i].log_name) {
      printf("Failed copying plugin log file name.\n");
      goto ErrorCleanup;
    }
    entry = cJSON_GetObjectItem(current_plugin, "label");
    if (entry) {
      if (entry->type != cJSON_String) {
        printf("Invalid plugin label in the config file.\n");
        goto ErrorCleanup;
      }
      plugins[i].label = strdup(entry->valuestring);
      if (!plugins[i].label) {
        printf("Failed copying plugin label.\n");
        goto ErrorCleanup;
      }
    }
    entry = cJSON_GetObjectItem(current_plugin, "thread_count");
    if (!entry || (entry->type != cJSON_Number)) {
      printf("Missing/invalid plugin thread_count in config.\n");
      goto ErrorCleanup;
    }
    plugins[i].thread_count = entry->valueint;
    entry = cJSON_GetObjectItem(current_plugin, "block_count");
    if (!entry || (entry->type != cJSON_Number)) {
      printf("Missing/invalid plugin block_count in config.\n");
      goto ErrorCleanup;
    }
    plugins[i].block_count = entry->valueint;
    entry = cJSON_GetObjectItem(current_plugin, "additional_info");
    if (entry) {
      plugins[i].additional_info = cJSON_PrintUnformatted(entry);
      if (!plugins[i].additional_info) {
        printf("Error copying additional info JSON.\n");
        goto ErrorCleanup;
      }
    }
    if (!ParseComputeUnitMask(current_plugin, plugins[i].compute_unit_mask)) {
      printf("Error processing the compute unit mask configuration.\n");
      goto ErrorCleanup;
    }
    entry = cJSON_GetObjectItem(current_plugin, "max_iterations");
    if (entry) {
      if (entry->type != cJSON_Number) {
        printf("Invalid plugin max_iterations in config.\n");
        goto ErrorCleanup;
      }
      // As with data_size, valuedouble provides a better range than valueint.
      plugins[i].max_iterations = entry->valuedouble;
      // We can't sync every iteration if some plugins will never reach the
      // barrier due to different numbers of iterations.
      if (config->sync_every_iteration) {
        printf("sync_every_iteration must be false if plugin-specific "
          "iteration counts are used.\n");
        goto ErrorCleanup;
      }
      // We can't sync every iteration if different plugins run different
      // numbers of iterations.
      config->sync_every_iteration = 0;
    } else {
      // Remember, 0 means unlimited, negative means unset.
      plugins[i].max_iterations = -1;
    }
    entry = cJSON_GetObjectItem(current_plugin, "max_time");
    if (entry) {
      if (entry->type != cJSON_Number) {
        printf("Invalid plugin max_time in config.\n");
        goto ErrorCleanup;
      }
      plugins[i].max_time = entry->valuedouble;
    } else {
      // As with max_iterations, negative means the value wasn't set.
      plugins[i].max_time = -1;
    }
    entry = cJSON_GetObjectItem(current_plugin, "release_time");
    if (entry) {
      if (entry->type != cJSON_Number) {
        printf("Invalid plugin release_time in config.\n");
        goto ErrorCleanup;
      }
      plugins[i].release_time = entry->valuedouble;
    } else {
      plugins[i].release_time = 0;
    }
    entry = cJSON_GetObjectItem(current_plugin, "cpu_core");
    if (entry) {
      if (entry->type != cJSON_Number) {
        printf("Invalid plugin CPU core in config.\n");
        goto ErrorCleanup;
      }
      plugins[i].cpu_core = entry->valueint;
    } else {
      plugins[i].cpu_core = USE_DEFAULT_CPU_CORE;
    }
    current_plugin = current_plugin->next;
  }
  config->plugins = plugins;
  config->plugin_count = plugin_count;
  return 1;
ErrorCleanup:
  // This won't free anything we didn't allocate, because we zero the entire
  // plugins array after allocating it.
  for (i = 0; i < plugin_count; i++) {
    if (plugins[i].filename) free(plugins[i].filename);
    if (plugins[i].log_name) free(plugins[i].log_name);
    if (plugins[i].additional_info) free(plugins[i].additional_info);
    if (plugins[i].label) free(plugins[i].label);
  }
  free(plugins);
  return 0;
}

GlobalConfiguration* ParseConfiguration(const char *config) {
  GlobalConfiguration *to_return = NULL;
  cJSON *root = NULL;
  cJSON *entry = NULL;
  int tmp;
  to_return = (GlobalConfiguration *) malloc(sizeof(*to_return));
  if (!to_return) {
    printf("Failed allocating config memory.\n");
    return NULL;
  }
  memset(to_return, 0, sizeof(*to_return));
  root = cJSON_Parse(config);
  if (!root) {
    printf("Failed parsing JSON.\n");
    free(to_return);
    return NULL;
  }
  if (!VerifyGlobalConfigKeys(root->child)) {
    goto ErrorCleanup;
  }
  // Begin reading the global settings values.
  entry = cJSON_GetObjectItem(root, "max_iterations");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Missing/invalid default max_iterations in config.\n");
    goto ErrorCleanup;
  }
  // Use valuedouble here, since valueint is just a double cast to an int
  // already. Casting valuedouble to a uint64_t will be just as good, and will
  // have a bigger range.
  to_return->max_iterations = entry->valuedouble;
  if (to_return->max_iterations < 0) {
    printf("Invalid(negative) default max_iterations in config.\n");
    goto ErrorCleanup;
  }
  entry = cJSON_GetObjectItem(root, "max_time");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Missing/invalid default max_time in config.\n");
    goto ErrorCleanup;
  }
  to_return->max_time = entry->valuedouble;
  entry = cJSON_GetObjectItem(root, "use_processes");
  if (entry) {
    tmp = entry->type;
    if ((tmp != cJSON_True) && (tmp != cJSON_False)) {
      printf("Invalid use_processes setting in config.\n");
      goto ErrorCleanup;
    }
    to_return->use_processes = tmp == cJSON_True;
  } else {
    to_return->use_processes = DEFAULT_USE_PROCESSES;
  }
  entry = cJSON_GetObjectItem(root, "gpu_device_id");
  if (!entry || (entry->type != cJSON_Number)) {
    printf("Missing/invalid GPU device ID in config.\n");
    goto ErrorCleanup;
  }
  to_return->gpu_device_id = entry->valueint;
  // Any string entries will be copied--we have to assume that freeing cJSON
  // and/or the config content will free them otherwise.
  entry = cJSON_GetObjectItem(root, "name");
  if (!entry || (entry->type != cJSON_String)) {
    printf("Missing scenario name in config.\n");
    goto ErrorCleanup;
  }
  to_return->scenario_name = strdup(entry->valuestring);
  if (!to_return->scenario_name) {
    printf("Failed allocating memory for the scenario name.\n");
    goto ErrorCleanup;
  }
  entry = cJSON_GetObjectItem(root, "base_result_directory");
  // Like the scenario_name entry, the result directory must also be copied.
  // However, it is optional so we'll copy the default if it's not present.
  if (entry) {
    if (entry->type != cJSON_String) {
      printf("Invalid base_result_directory in config.\n");
      goto ErrorCleanup;
    }
    to_return->base_result_directory = strdup(entry->valuestring);
  } else {
    to_return->base_result_directory = strdup(DEFAULT_BASE_RESULT_DIRECTORY);
  }
  if (!to_return->base_result_directory) {
    printf("Failed allocating memory for result path.\n");
    goto ErrorCleanup;
  }
  // The pin_cpus setting defaults to 0 (false)
  entry  = cJSON_GetObjectItem(root, "pin_cpus");
  if (entry) {
    tmp = entry->type;
    if ((tmp != cJSON_True) && (tmp != cJSON_False)) {
      printf("Invalid pin_cpus setting in config.\n");
      goto ErrorCleanup;
    }
    to_return->pin_cpus = tmp == cJSON_True;
  } else {
    to_return->pin_cpus = 0;
  }
  // The sync_every_iteration setting defaults to 0 (false). This MUST be
  // parsed before checking plugin-specific configs, to ensure that no plugin
  // has a specific iteration count while sync_every_iteration is true.
  entry = cJSON_GetObjectItem(root, "sync_every_iteration");
  if (entry) {
    tmp = entry->type;
    if ((tmp != cJSON_True) && (tmp != cJSON_False)) {
      printf("Invalid sync_every_iteration setting in config.\n");
      goto ErrorCleanup;
    }
    to_return->sync_every_iteration = tmp == cJSON_True;
  } else {
    to_return->sync_every_iteration = 0;
  }
  entry = cJSON_GetObjectItem(root, "do_warmup");
  if (entry) {
    tmp = entry->type;
    if ((tmp != cJSON_True) && (tmp != cJSON_False)) {
      printf("Invalid do_warmup setting in config.\n");
      goto ErrorCleanup;
    }
    to_return->do_warmup = tmp == cJSON_True;
  } else {
    to_return->do_warmup = 0;
  }
  entry = cJSON_GetObjectItem(root, "omit_block_times");
  if (entry) {
    tmp = entry->type;
    if ((tmp != cJSON_True) && (tmp != cJSON_False)) {
      printf("Invalid omit_block_times setting in config.\n");
      goto ErrorCleanup;
    }
    to_return->omit_block_times = tmp == cJSON_True;
  } else {
    to_return->omit_block_times = 0;
  }
  // Finally, parse the plugin list. Ensure that we've obtained a valid JSON
  // array for the plugins before calling ParsePluginList.
  entry = cJSON_GetObjectItem(root, "plugins");
  if (!entry || (entry->type != cJSON_Array) || !entry->child) {
    printf("Missing/invalid list of plugins in config.\n");
    goto ErrorCleanup;
  }
  entry = entry->child;
  if (!ParsePluginList(to_return, entry)) {
    goto ErrorCleanup;
  }
  // Clean up the JSON, we don't need it anymore since all the data was copied.
  cJSON_Delete(root);
  return to_return;
ErrorCleanup:
  if (to_return->base_result_directory) free(to_return->base_result_directory);
  if (to_return->scenario_name) free(to_return->scenario_name);
  free(to_return);
  cJSON_Delete(root);
  return NULL;
}

void FreeGlobalConfiguration(GlobalConfiguration *config) {
  int i;
  PluginConfiguration *plugins = config->plugins;
  for (i = 0; i < config->plugin_count; i++) {
    free(plugins[i].filename);
    free(plugins[i].log_name);
    if (plugins[i].additional_info) free(plugins[i].additional_info);
    if (plugins[i].label) free(plugins[i].label);
  }
  free(plugins);
  free(config->base_result_directory);
  free(config->scenario_name);
  memset(config, 0, sizeof(*config));
  free(config);
}

