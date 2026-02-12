/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/detail/doxygen.h>

#include <cstdint>

/**
 * @file
 * @brief Class definition for legate::ParallelPolicy
 */

namespace legate::mapping {

enum class TaskTarget : std::uint8_t;

}  // namespace legate::mapping

namespace legate::detail {

class Scope;
class Config;

}  // namespace legate::detail

namespace legate {

/**
 * @addtogroup tuning
 * @{
 */

/**
 * @brief Streaming modes for ParllelPolicy
 */
enum class StreamingMode : std::uint8_t {
  /**
   * @brief Disable Streaming.
   */
  OFF,
  /**
   * @brief Stream all the tasks created and submitted in the scope as a single
   * group.
   *
   * If any task in the scope is not 'Streamable', the runtime will throw a
   * `std::invalid_argument` exception.
   *
   * Users may use this mode when it is necessary that all tasks in the scope are
   * streamable, e.g., when the memory savings from streaming need to be maximized.
   * Users may also use this mode for debugging to catch non-streamable tasks.
   */
  STRICT,
  /**
   * @brief All tasks created and submitted in the scope need not stream as a
   * single group.
   *
   * This mode allows the runtime to break the tasks in a streaming scope into
   * multiple groups and add mapping fences between after every group. Users can
   * add this mode when it is not guaranteed or required that all tasks in the
   * scope are streamed as one group. This leaves some memory savings on the table
   * due to splitting the scope, but potentially allows third party library code to
   * be used inside a scope, for example.
   *
   */
  RELAXED,
};

/**
 * @brief A helper class that describes parallelization policies for tasks.
 *
 * A `ParallelPolicy` consists of knobs to control the parallelization policy for tasks in a given
 * scope. To change the parallelization policy of the scope, a new `Scope` must be created with a
 * `ParallelPolicy`. Currently, the `ParallelPolicy` class provides the following parameters:
 *
 *   - `streaming(StreamingMode)` (default: `OFF`): When the `streaming()` is not `OFF` in a scope,
 * the runtime executes the tasks in a streaming fashion. For example, if there are two tasks `T1`
 * and `T2` in the scope, the normal execution would run all parallel instances of `T1` before it
 * would move on to `T2`'s, whereas the streaming execution would alternative between `T1` and `T2`,
 *   launching a subset of parallel instances at a time that would fit to the memory. The
 *   granularity of tasks can be configured by the `overdecompose_factor()` (see below), and if the
 *   `overdecompose_factor()` is `1`, no streaming would happen even if the `streaming()` is `true`.
 *
 *   - `overdecompose_factor()` (default: `1`): When the value is greater than `1`, the
 *   auto-partitioner will over-decompose the stores when partitioning them; by default, the
 *   auto-partitioner creates `N` chunks in a store partition when there are `N` processors, but if
 *   the `overdecompose_factor()` is `k` in the scope, it would create `kN` chunks in the partition.
 */
class LEGATE_EXPORT ParallelPolicy {
 public:
  /**
   * @brief Sets the flag that indicates whether tasks in a given scope should be streamed.
   *
   * @param mode An enum of type StreamingMode that determines the mode of
   * streaming.
   *
   * @see StreamingMode.
   */
  ParallelPolicy& with_streaming(StreamingMode mode);
  /**
   * @brief Sets the over-decomposing factor.
   *
   * @param overdecompose_factor An over-decomposing factor.
   *
   * @see set_overdecompose_factor.
   */
  ParallelPolicy& with_overdecompose_factor(std::uint32_t overdecompose_factor);

  /**
   * @brief Sets partitioning threshold for target processor kind.
   *
   * This knob signals to the runtime that all stores whose volume is greater than
   * the threshold should be partitioned for the specified target processor kind.
   * Users may specify different thresholds for different task targets at the same
   * time.
   *
   * @param target Task target processor kind.
   * @param threshold partitioning threshold.
   */
  ParallelPolicy& with_partitioning_threshold(mapping::TaskTarget target, std::uint64_t threshold);

  /**
   * @brief Returns the streaming flag.
   *
   * @return true If the streaming is enabled.
   * @return false If the streaming is not enabled.
   */
  [[nodiscard]] bool streaming() const;

  /**
   * @brief Returns the streaming mode.
   *
   * @return enum value of type StreamingMode.
   */
  [[nodiscard]] StreamingMode streaming_mode() const;

  /**
   * @brief Returns the over-decomposing factor.
   *
   * @return The over-decomposing factor.
   */
  [[nodiscard]] std::uint32_t overdecompose_factor() const;

  /**
   * @brief Read the partitioning threshold for a give TaskTarget.
   *
   * @param target The target processor kind.
   *
   * @return Partitioning threshold.
   */
  [[nodiscard]] std::uint64_t partitioning_threshold(mapping::TaskTarget target) const;

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

  /**
   * @brief Initialize ParallelPolicy to system defined defaults.
   *
   * Initializes the ParallelPolicy members to default values defined by the system
   * configuration, which can be controlled via ``LEGATE_CONFIG`` environment
   * variable. Details:
   * - streaming_mode() : StreamingMode::OFF
   * - overdecompose_factor() : 1
   * - partitioning_threshold(CPU) : ``--cpu_chunk_size`` in ``LEGATE_CONFIG``
   * - partitioning_threshold(GPU) : ``gpu_chunk_size`` in ``LEGATE_CONFIG``
   * - partitioning_threshold(OMP) : ``omp_chunk_size`` in ``LEGATE_CONFIG``
   *
   * @note Legate runtime must be initialized before creating any ParallelPolicy
   * instance.
   */
  ParallelPolicy();

  class PrivateKey {
    PrivateKey() = default;

    friend class detail::Scope;
  };

  /**
   * @brief Initialize ParallelPolicy using config parameters.
   *
   * @param config Legate configuration.
   */
  ParallelPolicy(const detail::Config& config, PrivateKey);

 private:
  StreamingMode streaming_mode_{StreamingMode::OFF};
  std::uint32_t overdecompose_factor_{1};
  // these members are initialized to correct values in the constructor
  std::uint64_t cpu_partitioning_threshold_;
  std::uint64_t gpu_partitioning_threshold_;
  std::uint64_t omp_partitioning_threshold_;
};

/** @} */

}  // namespace legate

#include <legate/tuning/parallel_policy.inl>
