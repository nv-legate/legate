/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_array.h>
#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/scalar.h>
#include <legate/operation/detail/operation.h>
#include <legate/partitioning/constraint.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace legate {

class Scalar;

}  // namespace legate

namespace legate::detail {

class CommunicatorFactory;
class ConstraintSolver;
class Library;
class VariantInfo;

class TaskArrayArg {
 public:
  explicit TaskArrayArg(InternalSharedPtr<LogicalArray> _array);
  TaskArrayArg(InternalSharedPtr<LogicalArray> _array, std::optional<SymbolicPoint> _projection);
  [[nodiscard]] bool needs_flush() const;

  InternalSharedPtr<LogicalArray> array{};
  std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*> mapping{};
  std::optional<SymbolicPoint> projection{};
};

class Task : public Operation {
 protected:
  Task(const Library& library,
       const VariantInfo& variant_info,
       LocalTaskID task_id,
       std::uint64_t unique_id,
       std::int32_t priority,
       mapping::detail::Machine machine,
       bool can_inline_launch);

 public:
  void add_scalar_arg(InternalSharedPtr<Scalar> scalar);
  void set_concurrent(bool concurrent);
  void set_side_effect(bool has_side_effect);
  void throws_exception(bool can_throw_exception);
  void add_communicator(std::string_view name, bool bypass_signature_check = false);

  void record_scalar_output(InternalSharedPtr<LogicalStore> store);
  void record_unbound_output(InternalSharedPtr<LogicalStore> store);
  void record_scalar_reduction(InternalSharedPtr<LogicalStore> store,
                               GlobalRedopID legion_redop_id);

  void validate() override;

 protected:
  void launch_task_(Strategy* strategy);

 private:
  void inline_launch_() const;
  void legion_launch_(Strategy* strategy);

  void demux_scalar_stores_(const Legion::Future& result);
  void demux_scalar_stores_(const Legion::FutureMap& result, const Domain& launch_domain);

  // Calculate the return future size excluding the size of returned exception, which can only be
  // approximate
  [[nodiscard]] std::size_t calculate_future_size_() const;

 public:
  [[nodiscard]] std::string to_string(bool show_provenance) const override;
  [[nodiscard]] bool needs_flush() const override;
  [[nodiscard]] bool supports_replicated_write() const override;
  [[nodiscard]] bool can_throw_exception() const;
  [[nodiscard]] bool can_elide_device_ctx_sync() const;

  [[nodiscard]] const std::vector<InternalSharedPtr<Scalar>>& scalars() const;
  [[nodiscard]] const std::vector<TaskArrayArg>& inputs() const;
  [[nodiscard]] const std::vector<TaskArrayArg>& outputs() const;
  [[nodiscard]] const std::vector<TaskArrayArg>& reductions() const;
  [[nodiscard]] const std::vector<InternalSharedPtr<LogicalStore>>& scalar_outputs() const;
  [[nodiscard]] const std::vector<std::pair<InternalSharedPtr<LogicalStore>, GlobalRedopID>>&
  scalar_reductions() const;

  [[nodiscard]] const Library& library() const;
  [[nodiscard]] LocalTaskID local_task_id() const;

 protected:
  [[nodiscard]] const VariantInfo& variant_info_() const;

 private:
  std::reference_wrapper<const Library> library_;
  std::reference_wrapper<const VariantInfo> vinfo_;
  LocalTaskID task_id_{};
  bool concurrent_{};
  bool has_side_effect_{};
  bool can_throw_exception_{};
  bool can_elide_device_ctx_sync_{};
  std::vector<InternalSharedPtr<Scalar>> scalars_{};

 protected:
  std::vector<TaskArrayArg> inputs_{};
  std::vector<TaskArrayArg> outputs_{};
  std::vector<TaskArrayArg> reductions_{};
  std::vector<GlobalRedopID> reduction_ops_{};

 private:
  std::vector<InternalSharedPtr<LogicalStore>> unbound_outputs_{};
  std::vector<InternalSharedPtr<LogicalStore>> scalar_outputs_{};
  std::vector<std::pair<InternalSharedPtr<LogicalStore>, GlobalRedopID>> scalar_reductions_{};
  std::vector<std::reference_wrapper<CommunicatorFactory>> communicator_factories_{};
  bool can_inline_launch_{};
};

class AutoTask final : public Task {
 public:
  AutoTask(const Library& library,
           const VariantInfo& variant_info,
           LocalTaskID task_id,
           std::uint64_t unique_id,
           std::int32_t priority,
           mapping::detail::Machine machine);

  [[nodiscard]] const Variable* add_input(InternalSharedPtr<LogicalArray> array);
  [[nodiscard]] const Variable* add_output(InternalSharedPtr<LogicalArray> array);
  [[nodiscard]] const Variable* add_reduction(InternalSharedPtr<LogicalArray> array,
                                              std::int32_t redop_kind);

  void add_input(InternalSharedPtr<LogicalArray> array, const Variable* partition_symbol);
  void add_output(InternalSharedPtr<LogicalArray> array, const Variable* partition_symbol);
  void add_reduction(InternalSharedPtr<LogicalArray> array,
                     std::int32_t redop_kind,
                     const Variable* partition_symbol);

  [[nodiscard]] const Variable* find_or_declare_partition(
    const InternalSharedPtr<LogicalArray>& array);

  void add_constraint(InternalSharedPtr<Constraint> constraint,
                      bool bypass_signature_check = false);
  void add_to_solver(ConstraintSolver& solver) override;

  void validate() override;
  void launch(Strategy* strategy) override;

  [[nodiscard]] Kind kind() const override;

 private:
  void fixup_ranges_(Strategy& strategy);

  std::vector<InternalSharedPtr<Constraint>> constraints_{};
  std::vector<LogicalArray*> arrays_to_fixup_{};
};

class ManualTask final : public Task {
 public:
  ManualTask(const Library& library,
             const VariantInfo& variant_info,
             LocalTaskID task_id,
             const Domain& launch_domain,
             std::uint64_t unique_id,
             std::int32_t priority,
             mapping::detail::Machine machine);

  void add_input(const InternalSharedPtr<LogicalStore>& store);
  void add_input(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                 std::optional<SymbolicPoint> projection);
  void add_output(const InternalSharedPtr<LogicalStore>& store);
  void add_output(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                  std::optional<SymbolicPoint> projection);
  void add_reduction(const InternalSharedPtr<LogicalStore>& store, std::int32_t redop_kind);
  void add_reduction(const InternalSharedPtr<LogicalStorePartition>& store_partition,
                     std::int32_t redop_kind,
                     std::optional<SymbolicPoint> projection);

 private:
  void add_store_(std::vector<TaskArrayArg>& store_args,
                  const InternalSharedPtr<LogicalStore>& store,
                  InternalSharedPtr<Partition> partition,
                  std::optional<SymbolicPoint> projection = {});

 public:
  void launch() override;

  [[nodiscard]] Kind kind() const override;

 private:
  Strategy strategy_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/task.inl>
