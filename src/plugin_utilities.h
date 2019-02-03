// This file defines some utility functions that may be used across multiple
// plugins.
#ifndef PLUGIN_UTILITIES_H
#define PLUGIN_UTILITIES_H
#ifdef __cplusplus
extern "C" {
#endif

// Returns the current CPU time, in seconds.
double CurrentSeconds(void);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // PLUGIN_UTILITIES_H

