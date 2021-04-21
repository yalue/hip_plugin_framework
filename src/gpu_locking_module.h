// This file contains structs and definitions used when interfacing with the
// GPU locking module.

#ifndef GPU_LOCKING_MODULE_H
#define GPU_LOCKING_MODULE_H
#include <linux/ioctl.h>

#ifdef __cplusplus
extern "C" {
#endif

// Determines the number of locks that the module creates. The lock_id in the
// GPULockArgs struct must be less than this number.
#define GPU_LOCK_COUNT (4)

// A pointer to this struct is used as an argument for several ioctls to the
// module, including acquiring and releasing locks.
typedef struct {
  // The ID of the lock to acquire.
  uint32_t lock_id;
} GPULockArgs;

// A pointer to this struct is used as an argument to the ioctl for setting the
// deadline for the calling process. If no lock has been set, the process
// defaults to using a best-effort FIFO priority (expiring when it no longer
// has pending GPU work).  Processes with explicit deadlines always have higher
// priority than FIFO processes.
typedef struct {
  // The deadline to set. This must be a number of *nanoseconds relative to the
  // current time*.  Set this to 0 to unset task's deadline, making it revert
  // back to best-effort FIFO if it attempts to acquire the lock again.
  uint64_t deadline;
} SetDeadlineArgs;

// A pointer to this struct is used as an argument to the barrier-sync ioctl
// for waiting until the given number of processes have reached the barrier.
typedef struct {
  // The number of waiters to wait for (*including*) the caller. If any process
  // invokes the barrier-sync ioctl with this set incorrectly, then the ioctl
  // will immediately wake up all waiters and return an error.
  int count;
} BarrierSyncArgs;


// Send this ioctl to acquire the lock. This will return -EINTR *without*
// acquiring the lock, if a signal is received before the lock is acquired.
#define GPU_LOCK_ACQUIRE_IOC _IOW('h', 0xaa, GPULockArgs)

// Send this ioctl to release the lock, after it has been acquired. Returns an
// error if the requesting process doesn't hold the lock for the specified
// partition ID.
#define GPU_LOCK_RELEASE_IOC _IOW('h', 0xab, GPULockArgs)

// Send this ioctl to update a deadline associated with the calling task. If
// the "deadline" arg is set to 0, then this will make the task use the next
// best-effort FIFO priority. Otherwise, the arg specifies the relative number
// of nanoseconds to use as the deadline. This is applied even if the task is
// currently holding or waiting for each GPU lock, possibly causing it to be
// preempted or acquire the lock.
#define GPU_LOCK_SET_DEADLINE_IOC _IOW('h', 0xac, SetDeadlineArgs)

// This ioctl will return only after the number of processes specified in the
// "count" field of the arg have invoked it.
#define GPU_LOCK_BARRIER_SYNC_IOC _IOW('h', 0xb0, BarrierSyncArgs)

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // GPU_LOCKING_MODULE_H

