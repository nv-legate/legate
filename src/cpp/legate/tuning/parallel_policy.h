/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>

#include <cstdint>

/**
 * @file
 * @brief Class definition for legate::ParallelPolicy
 */

namespace legate {

/**
 * @addtogroup tuning
 * @{
 */

/**
 * @brief A helper class that describes parallelization policies for tasks.
 *
 * A `ParallelPolicy` consists of knobs to control the parallelization policy for tasks in a given
 * scope. To change the parallelization policy of the scope, a new `Scope` must be created with a
 * `ParallelPolicy`. Currently, the `ParallelPolicy` class provides the following parameters:
 *
 *   - `streaming()` (default: `false`): When the `streaming()` is `true` in a scope, the runtime
 *   executes the tasks in a streaming fashion. For example, if there are two tasks `T1` and `T2` in
 *   the scope, the normal execution would run all parallel instances of `T1` before it would move
 *   on to `T2`'s, whereas the streaming execution would alternative between `T1` and `T2`,
 *   launching a subset of parallel instances at a time that would fit to the memory. The
 *   granularity of tasks can be configured by the `overdecompose_factor()` (see below), and if the
 *   `overdecompose_factor()` is `1`, no streaming would happen even if the `streaming()` is `true`.
 *
 *   - `overdecompose_factor()` (default: `1`): When the value is greater than `1`, the
 *   auto-partitioner will over-decompose the stores when partitioning them; by default, the
 *   auto-partitioner creates `N` chunks in a store partition when there are `N` processors, but if
 *   the `overdecompose_factor()` is `k` in the scope, it would create `kN` chunks in the partition.
 */
class ParallelPolicy {
 public:
  /**
   * @brief Sets the flag that indicates whether tasks in a given scope should be streamed.
   *
   * @param streaming A boolean value to set to the streaming flag.
   */
  ParallelPolicy& with_streaming(bool streaming);
  /**
   * @brief Sets the over-decomposing factor.
   *
   * @param overdecompose_factor An over-decomposing factor.
   *
   * @see set_overdecompose_factor.
   */
  ParallelPolicy& with_overdecompose_factor(std::uint32_t overdecompose_factor);
  /**
   * @brief Returns the streaming flag.
   *
   * @return true If the streaming is enabled.
   * @return false If the streaming is not enabled.
   */
  [[nodiscard]] bool streaming() const;
  /**
   * @brief Returns the over-decomposing factor.
   *
   * @return The over-decomposing factor.
   */
  [[nodiscard]] std::uint32_t overdecompose_factor() const;
  /**
   * @brief Checks equality between `ParallelPolicy`s.
   *
   * @param other A `ParallelPolicy` to compare this with.
   *
   * @return true If `*this` is the same as `other`
   * @return false Otherwise.
   */
  [[nodiscard]] bool operator==(const ParallelPolicy& other) const;
  /**
   * @brief Checks inequality between `ParallelPolicy`s.
   *
   * @param other A `ParallelPolicy` to compare this with.
   *
   * @return true If `*this` is different from `other`
   * @return false Otherwise.
   */
  [[nodiscard]] bool operator!=(const ParallelPolicy& other) const;

 private:
  bool streaming_{false};
  std::uint32_t overdecompose_factor_{1};
};

/** @} */

}  // namespace legate

#include <legate/tuning/parallel_policy.inl>
