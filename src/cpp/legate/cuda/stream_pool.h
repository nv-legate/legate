/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/cuda/cuda.h>
#include <legate/utilities/detail/doxygen.h>

#include <optional>

using CUstream = struct CUstream_st*;

/**
 * @file
 * @brief Class definition for legate::cuda::StreamPool
 */

namespace legate::cuda {

/**
 * @addtogroup task
 * @{
 */

#if LEGATE_DEFINED(LEGATE_SILENCE_STREAM_POOL_DEPRECATION_PRIVATE)
#define LEGATE_STREAM_VIEW_DEPRECATED
#else
#define LEGATE_STREAM_VIEW_DEPRECATED \
  [[deprecated("since 24.11: provide your own implementation of this class")]]
#endif

/**
 * @brief A simple wrapper around CUDA streams to inject auxiliary features
 * @deprecated since 24.11: please provide your own implementation of this class
 *
 * When `LEGATE_SYNC_STREAM_VIEW` is set to 1, every `StreamView` synchronizes the CUDA stream
 * that it wraps when it is destroyed.
 */
class LEGATE_STREAM_VIEW_DEPRECATED StreamView {
 public:
  /**
   * @brief Creates a `StreamView` with a raw CUDA stream
   * @deprecated since 24.11: please provide your own implementation of this class
   *
   * @param stream Raw CUDA stream to wrap
   */
  explicit StreamView(CUstream stream);
  ~StreamView();

  StreamView(const StreamView&)            = delete;
  StreamView& operator=(const StreamView&) = delete;

  StreamView(StreamView&&) noexcept;
  StreamView& operator=(StreamView&&) noexcept;

  /**
   * @brief Unwraps the raw CUDA stream
   * @deprecated since 24.11: please provide your own implementation of this class
   *
   * @return Raw CUDA stream wrapped by the `StreamView`
   */
  // NOLINTNEXTLINE(google-explicit-constructor) implicit casting is intended here
  operator CUstream() const;

 private:
  bool valid_{};
  CUstream stream_{};
};

// We need this because in order to implement the deprecated functionality, we need to
// use... the deprecated functionality. And the compiler -- in their infinite wisdom -- just
// emit the same warnings for it anyways!
#if LEGATE_DEFINED(LEGATE_SILENCE_STREAM_POOL_DEPRECATION_PRIVATE)
#define LEGATE_STREAM_POOL_DEPRECATED
#else
#define LEGATE_STREAM_POOL_DEPRECATED \
  [[deprecated("since 24.11: use legate::TaskContext::get_task_stream() instead")]]
#endif

/**
 * @brief A stream pool
 * @deprecated since 24.11: use legate::TaskContext::get_task_stream() instead
 */
class LEGATE_STREAM_POOL_DEPRECATED StreamPool {
 public:
  /**
   * @brief Returns a `StreamView` in the pool
   * @deprecated since 24.11: use legate::TaskContext::get_task_stream() instead
   *
   * @return A `StreamView` object. Currently, all stream views returned from this pool are backed
   * by the same CUDA stream.
   */
  [[nodiscard]] StreamView get_stream();

  /**
   * @brief Returns a singleton stream pool
   * @deprecated since 24.11: use legate::TaskContext::get_task_stream() instead
   *
   * The stream pool is alive throughout the program execution.
   *
   * @return A `StreamPool` object
   */
  static StreamPool& get_stream_pool();

 private:
  // For now we keep only one stream in the pool
  std::optional<CUstream> cached_stream_{};
};

#undef LEGATE_SILENCE_STREAM_POOL_DEPRECATION_PRIVATE
#undef LEGATE_STREAM_POOL_DEPRECATED
#undef LEGATE_STREAM_VIEW_DEPRECATED

/** @} */

}  // namespace legate::cuda

#include <legate/cuda/stream_pool.inl>
