// This file implements the synchronization functions defined in
// multiprocess_sync.h.

#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include "multiprocess_sync.h"

// The internal state needed for barrier synchronization, to be held in shared
// memory.
typedef struct {
  // Maintains a count of the number of remaining processes.
  atomic_int processes_remaining;
  // Used in combination with local_sense to prevent multiple barriers from
  // interfering with each other.
  int sense;
  // This will be the number of processes which are sharing the barrier.
  int process_count;
} InternalSharedBarrier;

void* AllocateSharedBuffer(size_t size) {
  void *to_return = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS |
    MAP_SHARED, -1, 0);
  if (to_return == MAP_FAILED) return NULL;
  memset(to_return, 0, size);
  return to_return;
}

void FreeSharedBuffer(void *buffer, size_t size) {
  munmap(buffer, size);
}

int BarrierCreate(ProcessBarrier *b, int process_count) {
  InternalSharedBarrier *internal = NULL;
  b->internal_buffer = NULL;
  internal = (InternalSharedBarrier *) AllocateSharedBuffer(sizeof(*internal));
  if (!internal) return 0;
  internal->process_count = process_count;
  atomic_init(&(internal->processes_remaining), process_count);
  internal->sense = 0;
  b->internal_buffer = internal;
  return 1;
}

void BarrierDestroy(ProcessBarrier *b) {
  if (!b->internal_buffer) return;
  FreeSharedBuffer(b->internal_buffer, sizeof(InternalSharedBarrier));
  b->internal_buffer = NULL;
}

int BarrierWait(ProcessBarrier *b, int *local_sense) {
  volatile InternalSharedBarrier *internal =
    (InternalSharedBarrier *) b->internal_buffer;
  *local_sense = !(*local_sense);
  atomic_int value = atomic_fetch_sub(&(internal->processes_remaining), 1);
  if (value == 1) {
    // We were the last process to call atomic_fetch_sub, so reset the counter
    // and release the other processes by writing internal->sense.
    atomic_store(&(internal->processes_remaining), internal->process_count);
    internal->sense = *local_sense;
    return 1;
  }
  while (internal->sense != *local_sense) {
    sched_yield();
    continue;
  }
  return 1;
}

// The internal state needed to support the spin-based mutex. To be held in
// shared memory.
typedef struct {
  // This will be 0 if the mutex is free, and nonzero if it's being held.
  atomic_int v;
} InternalSharedMutex;

int MutexCreate(ProcessMutex *m) {
  InternalSharedMutex *internal = NULL;
  m->internal_buffer = NULL;
  internal = (InternalSharedMutex *) AllocateSharedBuffer(sizeof(*internal));
  if (!internal) return 0;
  atomic_init(&(internal->v), 0);
  m->internal_buffer = internal;
  return 1;
}

void MutexDestroy(ProcessMutex *m) {
  if (!m->internal_buffer) return;
  FreeSharedBuffer(m->internal_buffer, sizeof(InternalSharedMutex));
  m->internal_buffer = NULL;
}

int MutexAcquire(ProcessMutex *m) {
  volatile InternalSharedMutex *internal =
    (InternalSharedMutex *) m->internal_buffer;
  atomic_int expected;
  atomic_init(&expected, 0);
  // A simple test-and-test-and-set loop.
  while (1) {
    if (!internal->v) {
      if (atomic_compare_exchange_weak(&(internal->v), &expected, 1)) {
        break;
      }
    }
    // If we're here, that means that internal->v wasn't 0, so we'll need to
    // reset the expected value, since atomic_compare_exchange loads the
    // current value into expected on failure.
    atomic_store(&expected, 0);
    sched_yield();
  }
  return 1;
}

void MutexRelease(ProcessMutex *m) {
  volatile InternalSharedMutex *internal =
    (InternalSharedMutex *) m->internal_buffer;
  if (!internal->v) {
    printf("Error! Releasing a non-held mutex!\n");
    return;
  }
  atomic_store(&(internal->v), 0);
}

