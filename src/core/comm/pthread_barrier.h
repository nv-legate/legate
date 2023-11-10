/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#if !defined(_POSIX_BARRIERS) || (_POSIX_BARRIERS < 0)

#ifndef PTHREAD_BARRIER_H_
#define PTHREAD_BARRIER_H_

#include <pthread.h>

// This file provides a simple (and slow) implementation of pthread_barriers
// for Mac OS, as Mac does not implement pthread barriers, and the Legate
// implementation utilizes them when MPI is disabled.

#define CHECK_PTHREAD_CALL(...)       \
  do {                                \
    const int cpc_ret_ = __VA_ARGS__; \
    if (cpc_ret_) return cpc_ret_;    \
  } while (0)

using pthread_barrierattr_t = int;
struct pthread_barrier_t {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int count;
  int tripCount;
};

inline int pthread_barrier_init(pthread_barrier_t* barrier,
                                const pthread_barrierattr_t* /*attr*/,
                                unsigned int count)
{
  if (count == 0) { return -1; }
  CHECK_PTHREAD_CALL(pthread_mutex_init(&barrier->mutex, nullptr));
  if (const auto ret = pthread_cond_init(&barrier->cond, nullptr)) {
    CHECK_PTHREAD_CALL(pthread_mutex_destroy(&barrier->mutex));
    return ret;
  }
  barrier->tripCount = static_cast<decltype(barrier->tripCount)>(count);
  barrier->count     = 0;
  return 0;
}

inline int pthread_barrier_destroy(pthread_barrier_t* barrier)
{
  CHECK_PTHREAD_CALL(pthread_cond_destroy(&barrier->cond));
  CHECK_PTHREAD_CALL(pthread_mutex_destroy(&barrier->mutex));
  return 0;
}

inline int pthread_barrier_wait(pthread_barrier_t* barrier)
{
  CHECK_PTHREAD_CALL(pthread_mutex_lock(&barrier->mutex));
  ++(barrier->count);
  if (barrier->count >= barrier->tripCount) {
    barrier->count = 0;
    CHECK_PTHREAD_CALL(pthread_cond_broadcast(&barrier->cond));
    CHECK_PTHREAD_CALL(pthread_mutex_unlock(&barrier->mutex));
    return 1;
  }
  CHECK_PTHREAD_CALL(pthread_cond_wait(&barrier->cond, &(barrier->mutex)));
  CHECK_PTHREAD_CALL(pthread_mutex_unlock(&barrier->mutex));
  return 0;
}

#undef CHECK_PTHREAD_CALL

#endif  // PTHREAD_BARRIER_H_
#endif  // _POSIX_BARRIERS
