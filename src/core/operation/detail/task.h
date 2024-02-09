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

#include "core/data/detail/logical_array.h"
#include "core/data/detail/logical_store.h"
#include "core/data/detail/scalar.h"
#include "core/operation/detail/operation.h"
#include "core/partitioning/constraint.h"
#include "core/partitioning/detail/partitioner.h"
#include "core/utilities/internal_shared_ptr.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace legate {
class Scalar;
}  // namespace legate

namespace legate::detail {
class CommunicatorFactory;
struct ConstraintSolver;
class Library;
class Strategy;
class Runtime;

class Task : public Operation {
 protected:
  struct ArrayArg {
    explicit ArrayArg(InternalSharedPtr<LogicalArray> _array);
    ArrayArg(InternalSharedPtr<LogicalArray> _array, std::optional<SymbolicPoint> _projection);

    InternalSharedPtr<LogicalArray> array{};
    std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*> mapping{};
    std::optional<SymbolicPoint> projection{};
  };

  Task(const Library* library,
       int64_t task_id,
       uint64_t unique_id,
       mapping::detail::Machine&& machine);

 public:
  void add_scalar_arg(Scalar&& scalar);
  void set_concurrent(bool concurrent);
  void set_side_effect(bool has_side_effect);
  void throws_exception(bool can_throw_exception);
  void add_communicator(const std::string& name);

  void record_scalar_output(InternalSharedPtr<LogicalStore> store);
  void record_unbound_output(InternalSharedPtr<LogicalStore> store);
  void record_scalar_reduction(InternalSharedPtr<LogicalStore> store,
                               Legion::ReductionOpID legion_redop_id);

 protected:
  void launch_task(Strategy* strategy);

 private:
  void demux_scalar_stores(const Legion::Future& result);
  void demux_scalar_stores(const Legion::FutureMap& result, const Domain& launch_domain);

 public:
  [[nodiscard]] std::string to_string() const override;
  [[nodiscard]] bool always_flush() const override;
  [[nodiscard]] bool supports_replicated_write() const override;

 protected:
  const Library* library_{};
  int64_t task_id_{};
  bool concurrent_{};
  bool has_side_effect_{};
  bool can_throw_exception_{};
  std::vector<Scalar> scalars_{};
  std::vector<ArrayArg> inputs_{};
  std::vector<ArrayArg> outputs_{};
  std::vector<ArrayArg> reductions_{};
  std::vector<Legion::ReductionOpID> reduction_ops_{};
  std::vector<InternalSharedPtr<LogicalStore>> unbound_outputs_{};
  std::vector<InternalSharedPtr<LogicalStore>> scalar_outputs_{};
  std::vector<std::pair<InternalSharedPtr<LogicalStore>, Legion::ReductionOpID>>
    scalar_reductions_{};
  std::vector<CommunicatorFactory*> communicator_factories_{};
};

class AutoTask final : public Task {
 public:
  AutoTask(const Library* library,
           int64_t task_id,
           uint64_t unique_id,
           mapping::detail::Machine&& machine);

  [[nodiscard]] const Variable* add_input(InternalSharedPtr<LogicalArray> array);
  [[nodiscard]] const Variable* add_output(InternalSharedPtr<LogicalArray> array);
  [[nodiscard]] const Variable* add_reduction(InternalSharedPtr<LogicalArray> array, int32_t redop);

  void add_input(InternalSharedPtr<LogicalArray> array, const Variable* partition_symbol);
  void add_output(InternalSharedPtr<LogicalArray> array, const Variable* partition_symbol);
  void add_reduction(InternalSharedPtr<LogicalArray> array,
                     int32_t redop,
                     const Variable* partition_symbol);

  [[nodiscard]] const Variable* find_or_declare_partition(
    const InternalSharedPtr<LogicalArray>& array);

  void add_constraint(InternalSharedPtr<Constraint> constraint);
  void add_to_solver(ConstraintSolver& solver) override;

  void validate() override;
  void launch(Strategy* strategy) override;

 private:
  void fixup_ranges(Strategy& strategy);

  std::vector<InternalSharedPtr<Constraint>> constraints_{};
  std::vector<LogicalArray*> arrays_to_fixup_{};
};

class ManualTask final : public Task {
 public:
  ManualTask(const Library* library,
             int64_t task_id,
             const Domain& launch_domain,
             uint64_t unique_id,
             mapping::detail::Machine&& machine);

  void add_input(const InternalSharedPtr<LogicalStore>& store);
  void add_input(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                 std::optional<SymbolicPoint> projection);
  void add_output(const InternalSharedPtr<LogicalStore>& store);
  void add_output(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                  std::optional<SymbolicPoint> projection);
  void add_reduction(const InternalSharedPtr<LogicalStore>& store, Legion::ReductionOpID redop);
  void add_reduction(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                     Legion::ReductionOpID redop,
                     std::optional<SymbolicPoint> projection);

 private:
  void add_store(std::vector<ArrayArg>& store_args,
                 const InternalSharedPtr<LogicalStore>& store,
                 InternalSharedPtr<Partition> partition,
                 std::optional<SymbolicPoint> projection = {});

 public:
  void validate() override;
  void launch(Strategy* /*strategy*/) override;
  void launch();

  void add_to_solver(ConstraintSolver& solver) override;

 private:
  std::unique_ptr<Strategy> strategy_{};
};

}  // namespace legate::detail

#include "core/operation/detail/task.inl"
