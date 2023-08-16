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
#include "core/data/array.h"
#include "core/data/scalar.h"
#include "core/mapping/machine.h"

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
   * @brief Returns an input array of the task
   *
   * @param index Index of the array
   *
   * @return Array
   */
  Array input(uint32_t index) const;
  /**
   * @brief Returns all input arrays of the task
   *
   * @return Vector of arrays
   */
  std::vector<Array> inputs() const;
  /**
   * @brief Returns an output array of the task
   *
   * @param index Index of the array
   *
   * @return Array
   */
  Array output(uint32_t index) const;
  /**
   * @brief Returns all output arrays of the task
   *
   * @return Vector of arrays
   */
  std::vector<Array> outputs() const;
  /**
   * @brief Returns a reduction array of the task
   *
   * @param index Index of the array
   *
   * @return Array
   */
  Array reduction(uint32_t index) const;
  /**
   * @brief Returns all reduction arrays of the task
   *
   * @return Vector of arrays
   */
  std::vector<Array> reductions() const;
  /**
   * @brief Returns by-value arguments of the task
   *
   * @param index Index of the array
   *
   * @return Vector of scalar objects
   */
  const Scalar& scalar(uint32_t index) const;
  /**
   * @brief Returns by-value arguments of the task
   *
   * @return Vector of scalars
   */
  const std::vector<Scalar>& scalars() const;
  /**
   * @brief Returns communicators of the task
   *
   * If a task launch ends up emitting only a single point task, that task will not get passed a
   * communicator, even if one was requested at task launching time. Therefore, most tasks using
   * communicators should be prepared to handle the case where the returned vector is empty.
   *
   * @return Vector of communicator objects
   */
  std::vector<comm::Communicator> communicators() const;

 public:
  /**
   * @brief Returns the number of task's inputs
   *
   * @return Number of arrays
   */
  size_t num_inputs() const;
  /**
   * @brief Returns the number of task's outputs
   *
   * @return Number of arrays
   */
  size_t num_outputs() const;
  /**
   * @brief Returns the number of task's reductions
   *
   * @return Number of arrays
   */
  size_t num_reductions() const;

 public:
  /**
   * @brief Indicates whether the task is parallelized
   *
   * @return true The task is a single task
   * @return false The task is one in a set of multiple parallel tasks
   */
  bool is_single_task() const;
  /**
   * @brief Indicates whether the task is allowed to raise an exception
   *
   * @return true The task can raise an exception
   * @return false The task must not raise an exception
   */
  bool can_raise_exception() const;
  /**
   * @brief Returns the point of the task. A 0D point will be returned for a single task.
   *
   * @return The point of the task
   */
  DomainPoint get_task_index() const;
  /**
   * @brief Returns the task group's launch domain. A single task returns an empty domain
   *
   * @return The task group's launch domain
   */
  Domain get_launch_domain() const;

 public:
  mapping::Machine machine() const;

 public:
  const std::string& get_provenance() const;

 public:
  TaskContext(detail::TaskContext* impl);
  ~TaskContext();

 public:
  TaskContext(const TaskContext&);
  TaskContext& operator=(const TaskContext&);

 private:
  TaskContext(TaskContext&&)            = delete;
  TaskContext& operator=(TaskContext&&) = delete;

 public:
  detail::TaskContext* impl() const { return impl_; }

 private:
  detail::TaskContext* impl_{nullptr};
};

}  // namespace legate
