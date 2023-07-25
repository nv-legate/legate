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

#include <memory>

#include "core/data/detail/scalar.h"
#include "core/operation/detail/operation.h"
#include "core/partitioning/constraint.h"

namespace legate {
class Constraint;
class Scalar;
}  // namespace legate

namespace legate::detail {
class CommunicatorFactory;
class ConstraintSolver;
class Library;
class LogicalStore;
class LogicalStorePartition;
class Strategy;
class Runtime;

class Task : public Operation {
 protected:
  Task(const Library* library,
       int64_t task_id,
       uint64_t unique_id,
       mapping::detail::Machine&& machine);

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
  const Library* library_;
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
  AutoTask(const Library* library,
           int64_t task_id,
           uint64_t unique_id,
           mapping::detail::Machine&& machine);

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
  ManualTask(const Library* library,
             int64_t task_id,
             const Shape& launch_shape,
             uint64_t unique_id,
             mapping::detail::Machine&& machine);

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
