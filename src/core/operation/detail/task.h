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

#include "core/operation/detail/operation.h"
#include "core/partitioning/constraint.h"

namespace legate {
class Constraint;
class LibraryContext;
class Scalar;
}  // namespace legate

namespace legate::detail {
class CommunicatorFactory;
class ConstraintSolver;
class LogicalStore;
class LogicalStorePartition;
class Strategy;
class Runtime;

class Task : public Operation {
 protected:
  Task(const LibraryContext* library,
       int64_t task_id,
       uint64_t unique_id,
       mapping::MachineDesc&& machine);

 public:
  virtual ~Task() {}

 public:
  void add_scalar_arg(const Scalar& scalar);
  void add_scalar_arg(Scalar&& scalar);
  void set_concurrent(bool concurrent);
  void set_side_effect(bool has_side_effect);
  void throws_exception(bool can_throw_exception);
  void add_communicator(const std::string& name);

 public:
  virtual void launch(Strategy* strategy) override;

 private:
  void demux_scalar_stores(const Legion::Future& result);
  void demux_scalar_stores(const Legion::FutureMap& result, const Domain& launch_domain);

 public:
  std::string to_string() const override;

 protected:
  const LibraryContext* library_;
  int64_t task_id_;
  bool concurrent_{false};
  bool has_side_effect_{false};
  bool can_throw_exception_{false};
  std::vector<Scalar> scalars_{};
  std::vector<uint32_t> unbound_outputs_{};
  std::vector<uint32_t> scalar_outputs_{};
  std::vector<uint32_t> scalar_reductions_{};
  std::vector<CommunicatorFactory*> communicator_factories_{};
};

class AutoTask : public Task {
 private:
  friend class Runtime;
  AutoTask(const LibraryContext* library,
           int64_t task_id,
           uint64_t unique_id,
           mapping::MachineDesc&& machine);

 public:
  ~AutoTask() {}

 public:
  void add_input(std::shared_ptr<LogicalStore> store, const Variable* partition_symbol);
  void add_output(std::shared_ptr<LogicalStore> store, const Variable* partition_symbol);
  void add_reduction(std::shared_ptr<LogicalStore> store,
                     Legion::ReductionOpID redop,
                     const Variable* partition_symbol);

 private:
  void add_store(std::vector<StoreArg>& store_args,
                 std::shared_ptr<LogicalStore> store,
                 const Variable* partition_symbol);

 public:
  void add_constraint(std::unique_ptr<Constraint> constraint);
  void add_to_solver(ConstraintSolver& solver) override;

 public:
  void validate() override;

 private:
  std::vector<std::unique_ptr<Constraint>> constraints_{};
};

class ManualTask : public Task {
 private:
  friend class Runtime;
  ManualTask(const LibraryContext* library,
             int64_t task_id,
             const Shape& launch_shape,
             uint64_t unique_id,
             mapping::MachineDesc&& machine);

 public:
  ~ManualTask();

 public:
  void add_input(std::shared_ptr<LogicalStore> store);
  void add_input(std::shared_ptr<LogicalStorePartition> store_partition);
  void add_output(std::shared_ptr<LogicalStore> store);
  void add_output(std::shared_ptr<LogicalStorePartition> store_partition);
  void add_reduction(std::shared_ptr<LogicalStore> store, Legion::ReductionOpID redop);
  void add_reduction(std::shared_ptr<LogicalStorePartition> store_partition,
                     Legion::ReductionOpID redop);

 private:
  void add_store(std::vector<StoreArg>& store_args,
                 std::shared_ptr<LogicalStore> store,
                 std::shared_ptr<Partition> partition);

 public:
  void validate() override;
  void launch(Strategy* strategy) override;

 public:
  void add_to_solver(ConstraintSolver& solver) override;

 private:
  std::unique_ptr<Strategy> strategy_;
};

}  // namespace legate::detail
