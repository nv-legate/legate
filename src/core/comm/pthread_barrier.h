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

#pragma once

#include "core/utilities/abort.h"
#include "core/utilities/macros.h"

#include <cerrno>
#include <cstring>  // std::strerror
#include <pthread.h>
#include <unistd.h>  // _POSIX_BARRIERS

#define CHECK_PTHREAD_CALL(...)                                                      \
  do {                                                                               \
    const int cpc_ret_ = __VA_ARGS__;                                                \
    if (cpc_ret_) {                                                                  \
      if (!errno) {                                                                  \
        errno = cpc_ret_;                                                            \
      }                                                                              \
      legate::detail::log_legate().error()                                           \
        << std::strerror(errno) << " at "                                            \
        << __FILE__ ":" LegateStringize(__LINE__) ": " LegateStringize(__VA_ARGS__); \
      return cpc_ret_;                                                               \
    }                                                                                \
  } while (0)

#define CHECK_PTHREAD_CALL_V(...)                                                                \
  do {                                                                                           \
    const int cpc_ret_ = __VA_ARGS__;                                                            \
    if (cpc_ret_) {                                                                              \
      if (!errno) {                                                                              \
        errno = cpc_ret_;                                                                        \
      }                                                                                          \
      LEGATE_ABORT(std::strerror(errno)                                                          \
                   << " at "                                                                     \
                   << __FILE__ ":" LegateStringize(__LINE__) ": " LegateStringize(__VA_ARGS__)); \
    }                                                                                            \
  } while (0)

#if !defined(_POSIX_BARRIERS) || (_POSIX_BARRIERS < 0)

#include <atomic>

// This file provides a simple (and slow) implementation of pthread_barriers
// for Mac OS, as Mac does not implement pthread barriers, and the Legate
// implementation utilizes them when MPI is disabled.
using pthread_barrierattr_t = char;

struct pthread_barrier_t {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  unsigned int count;
  unsigned int limit;
  std::atomic<unsigned int> phase;
};

inline int pthread_barrier_init(pthread_barrier_t* barrier,
                                const pthread_barrierattr_t* /*attr*/,
                                unsigned int count)
{
  if (!count) {
    errno = EINVAL;
    return -1;
  }

  CHECK_PTHREAD_CALL(pthread_mutex_init(&barrier->mutex, nullptr));
  if (const auto ret = pthread_cond_init(&barrier->cond, nullptr)) {
    const auto errno_save = errno;

    static_cast<void>(pthread_mutex_destroy(&barrier->mutex));
    errno = errno_save;
    return ret;
  }
  barrier->limit = count;
  barrier->count = 0;
  barrier->phase = 0;
  return 0;
}

inline int pthread_barrier_destroy(pthread_barrier_t* barrier)
{
  CHECK_PTHREAD_CALL(pthread_cond_destroy(&barrier->cond));
  CHECK_PTHREAD_CALL(pthread_mutex_destroy(&barrier->mutex));
  return 0;
}

#define PTHREAD_BARRIER_SERIAL_THREAD 1

inline int pthread_barrier_wait(pthread_barrier_t* barrier)
{
  auto&& phase     = barrier->phase;
  const auto cond  = &barrier->cond;
  const auto mutex = &barrier->mutex;

  CHECK_PTHREAD_CALL(pthread_mutex_lock(mutex));
  barrier->count++;
  if (barrier->count == barrier->limit) {
    barrier->count = 0;
    phase.fetch_add(1, std::memory_order_release);
    CHECK_PTHREAD_CALL(pthread_cond_broadcast(cond));
    CHECK_PTHREAD_CALL(pthread_mutex_unlock(mutex));
    return PTHREAD_BARRIER_SERIAL_THREAD;
  }

  const auto old_phase = phase.load(std::memory_order_relaxed);

  do {
    CHECK_PTHREAD_CALL(pthread_cond_wait(cond, mutex));
  } while (old_phase == phase.load(std::memory_order_acquire));
  CHECK_PTHREAD_CALL(pthread_mutex_unlock(mutex));
  return 0;
}

#endif  // _POSIX_BARRIERS
