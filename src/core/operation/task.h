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

#include "core/data/logical_store.h"
#include "core/data/scalar.h"
#include "core/mapping/machine.h"
#include "core/partitioning/constraint.h"

/**
 * @file
 * @brief Class definitions for legate::AutoTask and legate::ManualTask
 */

namespace legate::detail {
class AutoTask;
class ManualTask;
}  // namespace legate::detail

namespace legate {

/**
 * @ingroup op
 * @brief A class for auto-parallelized task desciptors
 */
class AutoTask {
 public:
  /**
   * @brief Adds a store to the task as input
   *
   * Partitioning of the store is controlled by constraints on the partition symbol
   * associated with the store
   *
   * @param store A store to add to the task as input
   * @param partition_symbol A partition symbol for the store
   */
  void add_input(LogicalStore store, Variable partition_symbol);
  /**
   * @brief Adds a store to the task as output
   *
   * Partitioning of the store is controlled by constraints on the partition symbol
   * associated with the store
   *
   * @param store A store to add to the task as output
   * @param partition_symbol A partition symbol for the store
   */
  void add_output(LogicalStore store, Variable partition_symbol);
  /**
   * @brief Adds a store to the task for reductions
   *
   * Partitioning of the store is controlled by constraints on the partition symbol
   * associated with the store
   *
   * @param store A store to add to the task for reductions
   * @param redop ID of the reduction operator to use. The store's type must support the operator.
   * @param partition_symbol A partition symbol for the store
   */
  void add_reduction(LogicalStore store, ReductionOpKind redop, Variable partition_symbol);
  /**
   * @brief Adds a store to the task for reductions
   *
   * Partitioning of the store is controlled by constraints on the partition symbol
   * associated with the store
   *
   * @param store A store to add to the task for reductions
   * @param redop ID of the reduction operator to use. The store's type must support the operator.
   * @param partition_symbol A partition symbol for the store
   */
  void add_reduction(LogicalStore store, int32_t redop, Variable partition_symbol);
  /**
   * @brief Adds a by-value scalar argument to the task
   *
   * @param scalar A scalar to add to the task
   */
  void add_scalar_arg(const Scalar& scalar);
  /**
   * @brief Adds a by-value scalar argument to the task
   *
   * @param scalar A scalar to add to the task
   */
  void add_scalar_arg(Scalar&& scalar);
  /**
   * @brief Adds a partitioning constraint to the task
   *
   * @param constraint A partitioning constraint
   */
  void add_constraint(Constraint&& constraint);

 public:
  /**
   * @brief Finds or creates a partition symbol for the given store
   *
   * @param Store for which the partition symbol is queried
   *
   * @return The existing symbol if there is one for the store, a fresh symbol otherwise
   */
  Variable find_or_declare_partition(LogicalStore store);
  /**
   * @brief Declares partition symbol
   *
   * @return A new symbol that can be used when passing a store to an operation
   */
  Variable declare_partition();
  /**
   * @brief Returns the provenance information of this operation
   *
   * @return Provenance
   */
  const std::string& provenance() const;

 public:
  /**
   * @brief Sets whether the task needs a concurrent task launch.
   *
   * Any task with at least one communicator will implicitly use concurrent task launch, so this
   * method is to be used when the task needs a concurrent task launch for a reason unknown to
   * Legate.
   *
   * @param concurrent A boolean value indicating whether the task needs a concurrent task launch
   */
  void set_concurrent(bool concurrent);
  /**
   * @brief Sets whether the task has side effects or not.
   *
   * A task is assumed to be free of side effects by default if the task only has scalar arguments.
   *
   * @param has_side_effect A boolean value indicating whether the task has side effects
   */
  void set_side_effect(bool has_side_effect);
  /**
   * @brief Sets whether the task can throw an exception or not.
   *
   * @param can_throw_exception A boolean value indicating whether the task can throw an exception
   */
  void throws_exception(bool can_throw_exception);
  /**
   * @brief Requests a communicator for this task.
   *
   * @param name The name of the communicator to use for this task
   */
  void add_communicator(const std::string& name);

 public:
  AutoTask(AutoTask&&);
  AutoTask& operator=(AutoTask&&);

 private:
  AutoTask(const AutoTask&)            = delete;
  AutoTask& operator=(const AutoTask&) = delete;

 public:
  ~AutoTask();

 private:
  friend class Runtime;
  AutoTask(std::unique_ptr<detail::AutoTask> impl);
  std::unique_ptr<detail::AutoTask> impl_;
};

/**
 * @ingroup op
 * @brief A class for manually parallelized task descriptors
 */
class ManualTask {
 public:
  /**
   * @brief Adds a store to the task as input
   *
   * The store will be unpartitioned but broadcasted to all the tasks
   *
   * @param store A store to add to the task as input
   */
  void add_input(LogicalStore store);
  /**
   * @brief Adds a store partition to the task as input
   *
   * @param store_partition A store partition to add to the task as input
   */
  void add_input(LogicalStorePartition store_partition);
  /**
   * @brief Adds a store to the task as output
   *
   * The store will be unpartitioned but broadcasted to all the tasks
   *
   * @param store A store to add to the task as output
   */
  void add_output(LogicalStore store);
  /**
   * @brief Adds a store partition to the task as output
   *
   * @param store_partition A store partition to add to the task as output
   */
  void add_output(LogicalStorePartition store_partition);
  /**
   * @brief Adds a store to the task for reductions
   *
   * The store will be unpartitioned but broadcasted to all the tasks
   *
   * @param store A store to add to the task for reductions
   * @param redop ID of the reduction operator to use. The store's type must support the operator.
   */
  void add_reduction(LogicalStore store, ReductionOpKind redop);
  /**
   * @brief Adds a store to the task for reductions
   *
   * The store will be unpartitioned but broadcasted to all the tasks
   *
   * @param store A store to add to the task for reductions
   * @param redop ID of the reduction operator to use. The store's type must support the operator.
   */
  void add_reduction(LogicalStore store, int32_t redop);
  /**
   * @brief Adds a store partition to the task for reductions
   *
   * @param store_partition A store partition to add to the task for reductions
   * @param redop ID of the reduction operator to use. The store's type must support the operator.
   */
  void add_reduction(LogicalStorePartition store_partition, ReductionOpKind redop);
  /**
   * @brief Adds a store partition to the task for reductions
   *
   * @param store_partition A store partition to add to the task for reductions
   * @param redop ID of the reduction operator to use. The store's type must support the operator.
   */
  void add_reduction(LogicalStorePartition store_partition, int32_t redop);
  /**
   * @brief Adds a by-value scalar argument to the task
   *
   * @param scalar A scalar to add to the task
   */
  void add_scalar_arg(const Scalar& scalar);
  /**
   * @brief Adds a by-value scalar argument to the task
   *
   * @param scalar A scalar to add to the task
   */
  void add_scalar_arg(Scalar&& scalar);

 public:
  /**
   * @brief Returns the provenance information of this operation
   *
   * @return Provenance
   */
  const std::string& provenance() const;

 public:
  /**
   * @brief Sets whether the task needs a concurrent task launch.
   *
   * Any task with at least one communicator will implicitly use concurrent task launch, so this
   * method is to be used when the task needs a concurrent task launch for a reason unknown to
   * Legate.
   *
   * @param concurrent A boolean value indicating whether the task needs a concurrent task launch
   */
  void set_concurrent(bool concurrent);
  /**
   * @brief Sets whether the task has side effects or not.
   *
   * A task is assumed to be free of side effects by default if the task only has scalar arguments.
   *
   * @param has_side_effect A boolean value indicating whether the task has side effects
   */
  void set_side_effect(bool has_side_effect);
  /**
   * @brief Sets whether the task can throw an exception or not.
   *
   * @param can_throw_exception A boolean value indicating whether the task can throw an exception
   */
  void throws_exception(bool can_throw_exception);
  /**
   * @brief Requests a communicator for this task.
   *
   * @param name The name of the communicator to use for this task
   */
  void add_communicator(const std::string& name);

 public:
  ManualTask(ManualTask&&);
  ManualTask& operator=(ManualTask&&);

 private:
  ManualTask(const ManualTask&)            = delete;
  ManualTask& operator=(const ManualTask&) = delete;

 public:
  ~ManualTask();

 private:
  friend class Runtime;
  ManualTask(std::unique_ptr<detail::ManualTask> impl);
  std::unique_ptr<detail::ManualTask> impl_;
};

}  // namespace legate
