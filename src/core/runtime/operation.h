/* Copyright 2021 NVIDIA Corporation
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
#include <unordered_map>

#include "core/data/logical_store.h"
#include "core/mapping/machine.h"
#include "core/partitioning/constraint.h"
#include "legion.h"

/** @defgroup op Task and non-task operation descriptors
 */

/**
 * @file
 * @brief Class definitions for various operation kinds
 */

namespace legate {

class Constraint;
class ConstraintGraph;
class LibraryContext;
class Runtime;
class Scalar;
class Strategy;

namespace detail {

class LogicalStore;

}  // namespace detail

/**
 * @ingroup op
 * @brief A base class for all operation kinds
 */
class Operation {
 protected:
  using StoreArg = std::pair<detail::LogicalStore*, const Variable*>;
  Operation(LibraryContext* library, uint64_t unique_id, mapping::MachineDesc&& machine);

 public:
  virtual ~Operation() {}

 public:
  virtual void add_to_constraint_graph(ConstraintGraph& constraint_graph) const = 0;
  virtual void launch(Strategy* strategy)                                       = 0;
  virtual std::string to_string() const                                         = 0;

 public:
  /**
   * @brief Declares partition symbol
   *
   * @return A new symbol that can be used when passing a store to an operation
   */
  const Variable* declare_partition();
  detail::LogicalStore* find_store(const Variable* variable) const;

 public:
  /**
   * @brief Returns the machine of the scope in which this operation is issued
   *
   * @return The machine of the scope
   */
  const mapping::MachineDesc& machine() const { return machine_; }

  /**
   * @brief Returns the provenance information of this operation
   *
   * @return Provenance
   */
  const std::string& provenance() const { return provenance_; }

 protected:
  LibraryContext* library_;
  uint64_t unique_id_;

 protected:
  std::set<std::shared_ptr<detail::LogicalStore>> all_stores_{};
  std::vector<StoreArg> inputs_{};
  std::vector<StoreArg> outputs_{};
  std::vector<StoreArg> reductions_{};
  std::vector<Legion::ReductionOpID> reduction_ops_{};

 protected:
  uint32_t next_part_id_{0};
  std::vector<std::unique_ptr<Variable>> partition_symbols_{};
  std::map<const Variable, detail::LogicalStore*> store_mappings_{};
  std::string provenance_;
  mapping::MachineDesc machine_;
};

/**
 * @ingroup op
 * @brief A base class for tasks inherited by two kinds of task descriptors
 * (auto-parallelized and manually parallelized task descriptors)
 */
class Task : public Operation {
 protected:
  Task(LibraryContext* library,
       int64_t task_id,
       uint64_t unique_id,
       mapping::MachineDesc&& machine);

 public:
  virtual ~Task() {}

 public:
  /**
   * @brief Adds a by-value scalar argument to the task
   *
   * @param scalar A scalar to add to the task
   */
  void add_scalar_arg(const Scalar& scalar);

 public:
  virtual void launch(Strategy* strategy) override;

 private:
  void demux_scalar_stores(const Legion::Future& result);
  void demux_scalar_stores(const Legion::FutureMap& result, const Legion::Domain& launch_domain);

 public:
  std::string to_string() const override;

 protected:
  int64_t task_id_;
  std::vector<Scalar> scalars_{};
  std::vector<uint32_t> scalar_outputs_{};
  std::vector<uint32_t> scalar_reductions_{};
};

/**
 * @ingroup op
 * @brief A class for auto-parallelized task desciptors
 */
class AutoTask : public Task {
 public:
  friend class Runtime;
  AutoTask(LibraryContext* library,
           int64_t task_id,
           uint64_t unique_id,
           mapping::MachineDesc&& machine);

 public:
  ~AutoTask() {}

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
  void add_input(LogicalStore store, const Variable* partition_symbol);
  /**
   * @brief Adds a store to the task as output
   *
   * Partitioning of the store is controlled by constraints on the partition symbol
   * associated with the store
   *
   * @param store A store to add to the task as output
   * @param partition_symbol A partition symbol for the store
   */
  void add_output(LogicalStore store, const Variable* partition_symbol);
  /**
   * @brief Adds a store to the task for reductions
   *
   * Partitioning of the store is controlled by constraints on the partition symbol
   * associated with the store
   *
   * @param store A store to add to the task for reductions
   * @param redop ID of the reduction operator to use
   * @param partition_symbol A partition symbol for the store
   */
  void add_reduction(LogicalStore store,
                     Legion::ReductionOpID redop,
                     const Variable* partition_symbol);

 private:
  void add_store(std::vector<StoreArg>& store_args,
                 LogicalStore& store,
                 const Variable* partition_symbol);

 public:
  /**
   * @brief Adds a partitioning constraint to the task
   *
   * @param constraint A partitioning constraint
   */
  void add_constraint(std::unique_ptr<Constraint> constraint);
  void add_to_constraint_graph(ConstraintGraph& constraint_graph) const override;

 private:
  std::vector<std::unique_ptr<Constraint>> constraints_{};
};

/**
 * @ingroup op
 * @brief A class for manually parallelized task descriptors
 */
class ManualTask : public Task {
 private:
  friend class Runtime;
  ManualTask(LibraryContext* library,
             int64_t task_id,
             const Shape& launch_shape,
             uint64_t unique_id,
             mapping::MachineDesc&& machine);

 public:
  ~ManualTask();

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
   * @brief Adds a store to the task as output
   *
   * The store will be unpartitioned but broadcasted to all the tasks
   *
   * @param store A store to add to the task as output
   */
  void add_output(LogicalStore store);
  /**
   * @brief Adds a store to the task for reductions
   *
   * The store will be unpartitioned but broadcasted to all the tasks
   *
   * @param store A store to add to the task for reductions
   * @param redop ID of the reduction operator
   */
  void add_reduction(LogicalStore store, Legion::ReductionOpID redop);

 public:
  /**
   * @brief Adds a store partition to the task as input
   *
   * @param store_partition A store partition to add to the task as input
   */
  void add_input(LogicalStorePartition store_partition);
  /**
   * @brief Adds a store partition to the task as output
   *
   * @param store_partition A store partition to add to the task as output
   */
  void add_output(LogicalStorePartition store_partition);
  /**
   * @brief Adds a store partition to the task for reductions
   *
   * @param store_partition A store partition to add to the task for reductions
   * @param redop ID of the reduction operator
   */
  void add_reduction(LogicalStorePartition store_partition, Legion::ReductionOpID redop);

 private:
  void add_store(std::vector<StoreArg>& store_args,
                 const LogicalStore& store,
                 std::shared_ptr<Partition> partition);

 public:
  void launch(Strategy* strategy) override;

 public:
  void add_to_constraint_graph(ConstraintGraph& constraint_graph) const override;

 private:
  std::unique_ptr<Strategy> strategy_;
};

}  // namespace legate
