// This file defines some utility functions that may be used across multiple
// plugins.
#ifndef PLUGIN_UTILITIES_H
#define PLUGIN_UTILITIES_H
#include "plugin_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the current CPU time, in seconds.
double CurrentSeconds(void);

// A convenience function that copies the first dimension of 3-dimensional
// block and grid dimensions from the plugin's parameters. In other words, sets
// thread_count to block_dim[0] and block_count to grid_dim[0]. Returns 0 if
// any entry in block_dim or grid_dim other than the first has been set to a
// value other than 1.  Returns 1 on success.
int GetSingleBlockAndGridDimensions(InitializationParameters *params,
    int *thread_count, int *block_count);

// Like GetSingleBlockAndGridDimensions, but only checks and obtains the first
// dimension of params->block_dim.
int GetSingleBlockDimension(InitializationParameters *params,
    int *thread_count);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // PLUGIN_UTILITIES_H

