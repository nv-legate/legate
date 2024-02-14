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

#include "core/comm/communicator.h"
#include "core/data/physical_array.h"
#include "core/data/scalar.h"
#include "core/mapping/machine.h"

#include <optional>
#include <string>
#include <vector>

/**
 * @file
 * @brief Class definition for legate::TaskContext
 */

namespace legate::detail {
class TaskContext;
}  // namespace legate::detail

namespace legate {

/**
 * @ingroup task
 * @brief A task context that contains task arguments and communicators
 */
class TaskContext {
 public:
  /**
   * @brief Returns the global ID of the task
   *
   * @return The global task id
   */
  [[nodiscard]] std::int64_t task_id() const noexcept;
  /**
   * @brief Returns the Legate variant kind of the task
   *
   * @return The variant kind
   */
  [[nodiscard]] LegateVariantCode variant_kind() const noexcept;
  /**
   * @brief Returns an input array of the task
   *
   * @param index Index of the array
   *
   * @return Array
   */
  [[nodiscard]] PhysicalArray input(std::uint32_t index) const;
  /**
   * @brief Returns all input arrays of the task
   *
   * @return Vector of arrays
   */
  [[nodiscard]] std::vector<PhysicalArray> inputs() const;
  /**
   * @brief Returns an output array of the task
   *
   * @param index Index of the array
   *
   * @return Array
   */
  [[nodiscard]] PhysicalArray output(std::uint32_t index) const;
  /**
   * @brief Returns all output arrays of the task
   *
   * @return Vector of arrays
   */
  [[nodiscard]] std::vector<PhysicalArray> outputs() const;
  /**
   * @brief Returns a reduction array of the task
   *
   * @param index Index of the array
   *
   * @return Array
   */
  [[nodiscard]] PhysicalArray reduction(std::uint32_t index) const;
  /**
   * @brief Returns all reduction arrays of the task
   *
   * @return Vector of arrays
   */
  [[nodiscard]] std::vector<PhysicalArray> reductions() const;
  /**
   * @brief Returns a by-value argument of the task
   *
   * @param index Index of the scalar
   *
   * @return Scalar
   */
  [[nodiscard]] const Scalar& scalar(std::uint32_t index) const;
  /**
   * @brief Returns by-value arguments of the task
   *
   * @return Vector of scalars
   */
  [[nodiscard]] const std::vector<Scalar>& scalars() const;
  /**
   * @brief Returns a communicator of the task
   *
   * If a task launch ends up emitting only a single point task, that task will not get passed a
   * communicator, even if one was requested at task launching time. Therefore, tasks using
   * communicators should be prepared to handle the case where the returned vector is empty.
   *
   * @param index Index of the communicator
   *
   * @return Communicator
   */
  [[nodiscard]] comm::Communicator communicator(std::uint32_t index) const;
  /**
   * @brief Returns communicators of the task
   *
   * If a task launch ends up emitting only a single point task, that task will not get passed a
   * communicator, even if one was requested at task launching time. Therefore, most tasks using
   * communicators should be prepared to handle the case where the returned vector is empty.
   *
   * @return Vector of communicators
   */
  [[nodiscard]] std::vector<comm::Communicator> communicators() const;

  /**
   * @brief Returns the number of task's inputs
   *
   * @return Number of arrays
   */
  [[nodiscard]] std::size_t num_inputs() const;
  /**
   * @brief Returns the number of task's outputs
   *
   * @return Number of arrays
   */
  [[nodiscard]] std::size_t num_outputs() const;
  /**
   * @brief Returns the number of task's reductions
   *
   * @return Number of arrays
   */
  [[nodiscard]] std::size_t num_reductions() const;
  /**
   * @brief Returns the number of communicators
   *
   * @return Number of communicators
   */
  [[nodiscard]] std::size_t num_communicators() const;

  /**
   * @brief Indicates whether the task is parallelized
   *
   * @return true The task is a single task
   * @return false The task is one in a set of multiple parallel tasks
   */
  [[nodiscard]] bool is_single_task() const;
  /**
   * @brief Indicates whether the task is allowed to raise an exception
   *
   * @return true The task can raise an exception
   * @return false The task must not raise an exception
   */
  [[nodiscard]] bool can_raise_exception() const;
  /**
   * @brief Returns the point of the task. A 0D point will be returned for a single task.
   *
   * @return The point of the task
   */
  [[nodiscard]] DomainPoint get_task_index() const;
  /**
   * @brief Returns the task group's launch domain. A single task returns an empty domain
   *
   * @return The task group's launch domain
   */
  [[nodiscard]] Domain get_launch_domain() const;

  [[nodiscard]] mapping::Machine machine() const;

  [[nodiscard]] const std::string& get_provenance() const;

  [[nodiscard]] detail::TaskContext* impl() const;

  explicit TaskContext(detail::TaskContext* impl);

  TaskContext(const TaskContext&)            = default;
  TaskContext& operator=(const TaskContext&) = default;

  TaskContext(TaskContext&&)            = delete;
  TaskContext& operator=(TaskContext&&) = delete;

 private:
  detail::TaskContext* impl_{};
};

}  // namespace legate

#include "core/task/task_context.inl"
