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
#include "core/data/scalar.h"
#include "core/data/store.h"
#include "core/mapping/machine.h"

/**
 * @file
 * @brief Class definition for legate::TaskContext
 */

namespace legate::detail {
class TaskContext;
}  // namespace legate::detail

namespace legate {

class Store;
class Scalar;

/**
 * @ingroup task
 * @brief A task context that contains task arguments and communicators
 */
class TaskContext {
 public:
  /**
   * @brief Returns input stores of the task
   *
   * @return Vector of input stores
   */
  std::vector<Store>& inputs();
  /**
   * @brief Returns output stores of the task
   *
   * @return Vector of output stores
   */
  std::vector<Store>& outputs();
  /**
   * @brief Returns reduction stores of the task
   *
   * @return Vector of reduction stores
   */
  std::vector<Store>& reductions();
  /**
   * @brief Returns by-value arguments of the task
   *
   * @return Vector of scalar objects
   */
  std::vector<Scalar>& scalars();
  /**
   * @brief Returns communicators of the task
   *
   * If a task launch ends up emitting only a single point task, that task will not get passed a
   * communicator, even if one was requested at task launching time. Therefore, most tasks using
   * communicators should be prepared to handle the case where the returned vector is empty.
   *
   * @return Vector of communicator objects
   */
  std::vector<comm::Communicator>& communicators();

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
