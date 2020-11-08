// This file implements some spin-based userspace synchronization functions
// that use shared memory to work across multiple Linux processes.
#ifndef MULTIPROCESS_SYNC_H
#define MULTIPROCESS_SYNC_H
#ifdef __cplusplus
extern "C" {
#endif
#include <stdatomic.h>

// Holds the state of a barrier synchronization object that can be used between
// multiple processes or threads. Members of this struct shouldn't be modified
// directly.
typedef struct {
  // This will be a pointer to a buffer in shared memory. Will be NULL if the
  // barrier hasn't been created.
  void *internal_buffer;
} ProcessBarrier;

// Initializes the ProcessBarrier struct so that the given number of processes
// have to wait. Returns nonzero on success and 0 on error.
int BarrierCreate(ProcessBarrier *b, int process_count);

// Cleans up the given initialized process barrier. It is the caller's
// responsibility to ensure this isn't called while processes are still waiting
// on the barrier. Does nothing if the barrier wasn't created.
void BarrierDestroy(ProcessBarrier *b);

// Causes the process to wait on the given barrier. Internally, this will
// involve busy-waiting. Returns nonzero on success. local_sense *must* be a
// pointer to a non-shared local integer, initialized to 0, and then unchanged
// by the caller in subsequent calls to BarrierWait.
int BarrierWait(ProcessBarrier *b, int *local_sense);

// Holds the state of a mutex object that can be used between multiple
// processes or threads. Members of this struct shouldn't be modified directly.
typedef struct {
  // This will be a pointer to a buffer in shared memory. Will be NULL if the
  // mutex hasn't been created.
  void *internal_buffer;
} ProcessMutex;

// Initializes the ProcessMutex struct. The mutex is initialized as open.
int MutexCreate(ProcessMutex *m);

// Cleans up the mutex. It's the caller's responsibility to ensure that there
// aren't any waiters remaining. Does nothing if the mutex wasn't created.
void MutexDestroy(ProcessMutex *m);

// Acquires the ProcessMutex. Returns 0 if an error occurs. Uses *spinning* so
// do not use it for long-running tasks. This provides no guarantees about
// ordering, fairness, or efficiency. It only does one thing: ensure that only
// one process acquires the mutex at a time.
int MutexAcquire(ProcessMutex *m);

// Releases the mutex.
void MutexRelease(ProcessMutex *m);

// Allocates a private shared memory buffer containing the given number of
// bytes. Can be freed by using FreeSharedBuffer. Returns NULL on error.
// Initializes the buffer to contain 0.
void *AllocateSharedBuffer(size_t size);

// Frees a shared buffer returned by AllocateSharedBuffer.
void FreeSharedBuffer(void *buffer, size_t size);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // MULTIPROCESS_SYNC_H
