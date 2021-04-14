#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "plugin_interface.h"
#include "plugin_utilities.h"

double CurrentSeconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    printf("Error getting time.\n");
    exit(1);
  }
  return ((double) ts.tv_sec) + (((double) ts.tv_nsec) / 1e9);
}

int GetSingleBlockAndGridDimensions(InitializationParameters *params,
    int *thread_count, int *block_count) {
  int a, b;
  if ((params->block_dim[1] != 1) || (params->block_dim[2] != 1)) {
    printf("Expected 1-D block dimensions, but got [%d, %d, %d]\n",
      params->block_dim[0], params->block_dim[1], params->block_dim[2]);
    return 0;
  }
  if ((params->grid_dim[1] != 1) || (params->grid_dim[2] != 1)) {
    printf("Expected 1-D grid dimensions, but got [%d, %d, %d]\n",
      params->grid_dim[0], params->grid_dim[1], params->grid_dim[2]);
    return 0;
  }
  a = params->block_dim[0];
  if ((a < 1) || (a > 1024)) {
    printf("Invalid number of threads in a block: %d\n", a);
    return 0;
  }
  b = params->grid_dim[0];
  if (b < 1) {
    printf("Invalid number of blocks: %d\n", b);
  }
  *thread_count = a;
  *block_count = b;
  return 1;
}

int GetSingleBlockDimension(InitializationParameters *params,
    int *thread_count) {
  int x, y, z;
  x = params->block_dim[0];
  y = params->block_dim[1];
  z = params->block_dim[2];
  if ((y != 1) || (z != 1)) {
    printf("Expected 1-D block dimensions, but got [%d, %d, %d]\n", x, y, z);
    return 0;
  }
  *thread_count = x;
  return 1;
}
