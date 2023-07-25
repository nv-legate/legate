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

#include <memory>

#include "core/data/scalar.h"
#include "core/mapping/store.h"

/**
 * @file
 * @brief Class definitions for operations and stores used in mapping
 */

namespace legate::mapping {

namespace detail {
class Task;
}  // namespace detail

/**
 * @ingroup mapping
 * @brief A metadata class for tasks
 */
class Task {
 public:
  /**
   * @brief Returns the task id
   *
   * @return Task id
   */
  int64_t task_id() const;

 public:
  /**
   * @brief Returns metadata for the task's input stores
   *
   * @return Vector of store metadata objects
   */
  std::vector<Store> inputs() const;
  /**
   * @brief Returns metadata for the task's output stores
   *
   * @return Vector of store metadata objects
   */
  std::vector<Store> outputs() const;
  /**
   * @brief Returns metadata for the task's reduction stores
   *
   * @return Vector of store metadata objects
   */
  std::vector<Store> reductions() const;
  /**
   * @brief Returns the vector of the task's by-value arguments. Unlike `mapping::Store`
   * objects that have no access to data in the stores, the returned `Scalar` objects
   * contain valid arguments to the task
   *
   * @return Vector of `Scalar` objects
   */
  const std::vector<Scalar>& scalars() const;

 public:
  /**
   * @brief Returns metadata for the task's input store
   *
   * @param index Index of the input store
   *
   * @return Store metadata object
   */
  Store input(uint32_t index) const;
  /**
   * @brief Returns metadata for the task's output store
   *
   * @param index Index of the output store
   *
   * @return Store metadata object
   */
  Store output(uint32_t index) const;
  /**
   * @brief Returns metadata for the task's reduction store
   *
   * @param index Index of the reduction store
   *
   * @return Store metadata object
   */
  Store reduction(uint32_t index) const;

 public:
  /**
   * @brief Returns the number of task's inputs
   *
   * @return Number of stores
   */
  size_t num_inputs() const;
  /**
   * @brief Returns the number of task's outputs
   *
   * @return Number of stores
   */
  size_t num_outputs() const;
  /**
   * @brief Returns the number of task's reductions
   *
   * @return Number of stores
   */
  size_t num_reductions() const;

 public:
  Task(detail::Task* impl_);
  ~Task();

 private:
  Task(const Task&)            = delete;
  Task& operator=(const Task&) = delete;
  Task(Task&&)                 = delete;
  Task& operator=(Task&&)      = delete;

 private:
  detail::Task* impl_{nullptr};
};

}  // namespace legate::mapping
