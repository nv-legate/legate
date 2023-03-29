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
#include "core/partitioning/constraint.h"
#include "legion.h"

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

class Operation {
 protected:
  using StoreArg = std::pair<detail::LogicalStore*, const Variable*>;
  Operation(LibraryContext* library, uint64_t unique_id, int64_t mapper_id);

 public:
  virtual ~Operation() {}

 public:
  virtual void add_to_constraint_graph(ConstraintGraph& constraint_graph) const = 0;
  virtual void launch(Strategy* strategy)                                       = 0;
  virtual std::string to_string() const                                         = 0;

 public:
  const Variable* declare_partition();
  detail::LogicalStore* find_store(const Variable* variable) const;

 protected:
  LibraryContext* library_;
  uint64_t unique_id_;
  int64_t mapper_id_;

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
};

class Task : public Operation {
 protected:
  Task(LibraryContext* library, int64_t task_id, uint64_t unique_id, int64_t mapper_id);

 public:
  virtual ~Task() {}

 public:
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

class AutoTask : public Task {
 public:
  friend class Runtime;
  AutoTask(LibraryContext* library, int64_t task_id, uint64_t unique_id, int64_t mapper_id);

 public:
  ~AutoTask() {}

 public:
  void add_input(LogicalStore store, const Variable* partition_symbol);
  void add_output(LogicalStore store, const Variable* partition_symbol);
  void add_reduction(LogicalStore store,
                     Legion::ReductionOpID redop,
                     const Variable* partition_symbol);

 private:
  void add_store(std::vector<StoreArg>& store_args,
                 LogicalStore& store,
                 const Variable* partition_symbol);

 public:
  void add_constraint(std::unique_ptr<Constraint> constraint);
  void add_to_constraint_graph(ConstraintGraph& constraint_graph) const override;

 private:
  std::vector<std::unique_ptr<Constraint>> constraints_{};
};

class ManualTask : public Task {
 private:
  friend class Runtime;
  ManualTask(LibraryContext* library,
             int64_t task_id,
             const Shape& launch_shape,
             uint64_t unique_id,
             int64_t mapper_id);

 public:
  ~ManualTask();

 public:
  void add_input(LogicalStore store);
  void add_output(LogicalStore store);
  void add_reduction(LogicalStore store, Legion::ReductionOpID redop);

 public:
  void add_input(LogicalStorePartition store_partition);
  void add_output(LogicalStorePartition store_partition);
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
