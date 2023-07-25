/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
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
