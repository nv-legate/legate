/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/array_kind.h>
#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/shape.h>
#include <legate/data/detail/user_storage_tracker.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/operation/projection.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

namespace legate::detail {

class PhysicalArray;
class AutoTask;
class ConstraintSolver;
class ListLogicalArray;
class TaskReturnLayoutForUnpack;
class Variable;

class LogicalArray {
 public:
  LogicalArray(const LogicalArray&)                                   = default;
  LogicalArray(LogicalArray&&)                                        = delete;
  LogicalArray& operator=(const LogicalArray&)                        = default;
  LogicalArray& operator=(LogicalArray&&)                             = delete;
  LogicalArray()                                                      = default;
  virtual ~LogicalArray()                                             = default;
  [[nodiscard]] virtual std::uint32_t dim() const                     = 0;
  [[nodiscard]] virtual ArrayKind kind() const                        = 0;
  [[nodiscard]] virtual const InternalSharedPtr<Type>& type() const   = 0;
  [[nodiscard]] virtual const InternalSharedPtr<Shape>& shape() const = 0;
  [[nodiscard]] virtual std::size_t volume() const                    = 0;
  [[nodiscard]] virtual bool unbound() const                          = 0;
  [[nodiscard]] virtual bool nullable() const                         = 0;
  [[nodiscard]] virtual bool nested() const                           = 0;
  [[nodiscard]] virtual std::uint32_t num_children() const            = 0;
  [[nodiscard]] virtual bool is_mapped() const                        = 0;

  [[nodiscard]] virtual InternalSharedPtr<LogicalArray> promote(std::int32_t extra_dim,
                                                                std::size_t dim_size) const     = 0;
  [[nodiscard]] virtual InternalSharedPtr<LogicalArray> project(std::int32_t dim,
                                                                std::int64_t index) const       = 0;
  [[nodiscard]] virtual InternalSharedPtr<LogicalArray> slice(std::int32_t dim, Slice sl) const = 0;
  [[nodiscard]] virtual InternalSharedPtr<LogicalArray> transpose(
    SmallVector<std::int32_t, LEGATE_MAX_DIM> axes) const = 0;
  [[nodiscard]] virtual InternalSharedPtr<LogicalArray> delinearize(
    std::int32_t dim, SmallVector<std::uint64_t, LEGATE_MAX_DIM> sizes) const = 0;

  [[nodiscard]] virtual const InternalSharedPtr<LogicalStore>& data() const;
  [[nodiscard]] virtual const InternalSharedPtr<LogicalStore>& null_mask() const = 0;
  [[nodiscard]] virtual InternalSharedPtr<PhysicalArray> get_physical_array(
    legate::mapping::StoreTarget target, bool ignore_future_mutability) const            = 0;
  [[nodiscard]] virtual InternalSharedPtr<LogicalArray> child(std::uint32_t index) const = 0;
  [[nodiscard]] virtual const InternalSharedPtr<LogicalStore>& primary_store() const     = 0;

  virtual void record_scalar_or_unbound_outputs(AutoTask* task) const              = 0;
  virtual void record_scalar_reductions(AutoTask* task, GlobalRedopID redop) const = 0;

  virtual void generate_constraints(
    AutoTask* task,
    std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
    const Variable* partition_symbol) const = 0;

  [[nodiscard]] virtual ArrayAnalyzable to_launcher_arg(
    const std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
    const Strategy& strategy,
    const Domain& launch_domain,
    const std::optional<SymbolicPoint>& projection,
    Legion::PrivilegeMode privilege,
    GlobalRedopID redop) const = 0;
  [[nodiscard]] virtual ArrayAnalyzable to_launcher_arg_for_fixup(
    const Domain& launch_domain, Legion::PrivilegeMode privilege) const = 0;

  [[nodiscard]] static InternalSharedPtr<LogicalArray> from_store(
    InternalSharedPtr<LogicalStore> store);
  [[nodiscard]] bool needs_flush() const;

  virtual void collect_storage_trackers(SmallVector<UserStorageTracker>& trackers) const = 0;
  virtual void calculate_pack_size(TaskReturnLayoutForUnpack* layout) const              = 0;
};

class BaseLogicalArray final : public LogicalArray {
 public:
  explicit BaseLogicalArray(
    InternalSharedPtr<LogicalStore> data,
    std::optional<InternalSharedPtr<LogicalStore>> null_mask = std::nullopt);

  [[nodiscard]] std::uint32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
  [[nodiscard]] const InternalSharedPtr<Type>& type() const override;
  [[nodiscard]] const InternalSharedPtr<Shape>& shape() const override;
  [[nodiscard]] std::size_t volume() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] std::uint32_t num_children() const override;
  [[nodiscard]] bool is_mapped() const override;

  [[nodiscard]] InternalSharedPtr<LogicalArray> promote(std::int32_t extra_dim,
                                                        std::size_t dim_size) const override;
  [[nodiscard]] InternalSharedPtr<LogicalArray> project(std::int32_t dim,
                                                        std::int64_t index) const override;
  [[nodiscard]] InternalSharedPtr<LogicalArray> slice(std::int32_t dim, Slice sl) const override;
  [[nodiscard]] InternalSharedPtr<LogicalArray> transpose(
    SmallVector<std::int32_t, LEGATE_MAX_DIM> axes) const override;
  [[nodiscard]] InternalSharedPtr<LogicalArray> delinearize(
    std::int32_t dim, SmallVector<std::uint64_t, LEGATE_MAX_DIM> sizes) const override;

  [[nodiscard]] const InternalSharedPtr<LogicalStore>& data() const override;
  [[nodiscard]] const InternalSharedPtr<LogicalStore>& null_mask() const override;
  [[nodiscard]] InternalSharedPtr<PhysicalArray> get_physical_array(
    legate::mapping::StoreTarget target, bool ignore_future_mutability) const override;
  [[nodiscard]] InternalSharedPtr<BasePhysicalArray> get_base_physical_array(
    legate::mapping::StoreTarget target, bool ignore_future_mutability) const;
  [[nodiscard]] InternalSharedPtr<LogicalArray> child(std::uint32_t index) const override;
  [[nodiscard]] const InternalSharedPtr<LogicalStore>& primary_store() const override;

  void record_scalar_or_unbound_outputs(AutoTask* task) const override;
  void record_scalar_reductions(AutoTask* task, GlobalRedopID redop) const override;

  void generate_constraints(
    AutoTask* task,
    std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
    const Variable* partition_symbol) const override;

  [[nodiscard]] ArrayAnalyzable to_launcher_arg(
    const std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
    const Strategy& strategy,
    const Domain& launch_domain,
    const std::optional<SymbolicPoint>& projection,
    Legion::PrivilegeMode privilege,
    GlobalRedopID redop) const override;
  [[nodiscard]] ArrayAnalyzable to_launcher_arg_for_fixup(
    const Domain& launch_domain, Legion::PrivilegeMode privilege) const override;

  void collect_storage_trackers(SmallVector<UserStorageTracker>& trackers) const override;
  void calculate_pack_size(TaskReturnLayoutForUnpack* layout) const override;

 private:
  InternalSharedPtr<LogicalStore> data_{};
  std::optional<InternalSharedPtr<LogicalStore>> null_mask_{};
};

class ListLogicalArray final : public LogicalArray {
 public:
  ListLogicalArray(InternalSharedPtr<Type> type,
                   InternalSharedPtr<BaseLogicalArray> descriptor,
                   InternalSharedPtr<LogicalArray> vardata);

  [[nodiscard]] std::uint32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
  [[nodiscard]] const InternalSharedPtr<Type>& type() const override;
  [[nodiscard]] const InternalSharedPtr<Shape>& shape() const override;
  [[nodiscard]] std::size_t volume() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] std::uint32_t num_children() const override;
  [[nodiscard]] bool is_mapped() const override;

  [[nodiscard]] InternalSharedPtr<LogicalArray> promote(std::int32_t extra_dim,
                                                        std::size_t dim_size) const override;
  [[nodiscard]] InternalSharedPtr<LogicalArray> project(std::int32_t dim,
                                                        std::int64_t index) const override;
  [[nodiscard]] InternalSharedPtr<LogicalArray> slice(std::int32_t dim, Slice sl) const override;
  [[nodiscard]] InternalSharedPtr<LogicalArray> transpose(
    SmallVector<std::int32_t, LEGATE_MAX_DIM> axes) const override;
  [[nodiscard]] InternalSharedPtr<LogicalArray> delinearize(
    std::int32_t dim, SmallVector<std::uint64_t, LEGATE_MAX_DIM> sizes) const override;

  [[nodiscard]] const InternalSharedPtr<LogicalStore>& null_mask() const override;
  [[nodiscard]] InternalSharedPtr<PhysicalArray> get_physical_array(
    legate::mapping::StoreTarget target, bool ignore_future_mutability) const override;
  [[nodiscard]] InternalSharedPtr<LogicalArray> child(std::uint32_t index) const override;
  [[nodiscard]] const InternalSharedPtr<LogicalStore>& primary_store() const override;
  [[nodiscard]] const InternalSharedPtr<BaseLogicalArray>& descriptor() const;
  [[nodiscard]] const InternalSharedPtr<LogicalArray>& vardata() const;

  void record_scalar_or_unbound_outputs(AutoTask* task) const override;
  void record_scalar_reductions(AutoTask* task, GlobalRedopID redop) const override;

  void generate_constraints(
    AutoTask* task,
    std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
    const Variable* partition_symbol) const override;

  [[nodiscard]] ArrayAnalyzable to_launcher_arg(
    const std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
    const Strategy& strategy,
    const Domain& launch_domain,
    const std::optional<SymbolicPoint>& projection,
    Legion::PrivilegeMode privilege,
    GlobalRedopID redop) const override;
  [[nodiscard]] ArrayAnalyzable to_launcher_arg_for_fixup(
    const Domain& launch_domain, Legion::PrivilegeMode privilege) const override;

  void collect_storage_trackers(SmallVector<UserStorageTracker>& trackers) const override;
  void calculate_pack_size(TaskReturnLayoutForUnpack* layout) const override;

 private:
  InternalSharedPtr<Type> type_{};
  InternalSharedPtr<BaseLogicalArray> descriptor_{};
  InternalSharedPtr<LogicalArray> vardata_{};
};

class StructLogicalArray final : public LogicalArray {
 public:
  StructLogicalArray(InternalSharedPtr<Type> type,
                     std::optional<InternalSharedPtr<LogicalStore>> null_mask,
                     SmallVector<InternalSharedPtr<LogicalArray>>&& fields);

  [[nodiscard]] std::uint32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
  [[nodiscard]] const InternalSharedPtr<Type>& type() const override;
  [[nodiscard]] const InternalSharedPtr<Shape>& shape() const override;
  [[nodiscard]] std::size_t volume() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] std::uint32_t num_children() const override;
  [[nodiscard]] bool is_mapped() const override;

  [[nodiscard]] InternalSharedPtr<LogicalArray> promote(std::int32_t extra_dim,
                                                        std::size_t dim_size) const override;
  [[nodiscard]] InternalSharedPtr<LogicalArray> project(std::int32_t dim,
                                                        std::int64_t index) const override;
  [[nodiscard]] InternalSharedPtr<LogicalArray> slice(std::int32_t dim, Slice sl) const override;
  [[nodiscard]] InternalSharedPtr<LogicalArray> transpose(
    SmallVector<std::int32_t, LEGATE_MAX_DIM> axes) const override;
  [[nodiscard]] InternalSharedPtr<LogicalArray> delinearize(
    std::int32_t dim, SmallVector<std::uint64_t, LEGATE_MAX_DIM> sizes) const override;

  [[nodiscard]] const InternalSharedPtr<LogicalStore>& null_mask() const override;
  [[nodiscard]] InternalSharedPtr<PhysicalArray> get_physical_array(
    legate::mapping::StoreTarget target, bool ignore_future_mutability) const override;
  [[nodiscard]] InternalSharedPtr<LogicalArray> child(std::uint32_t index) const override;
  [[nodiscard]] const InternalSharedPtr<LogicalStore>& primary_store() const override;
  [[nodiscard]] Span<const InternalSharedPtr<LogicalArray>> fields() const;

  void record_scalar_or_unbound_outputs(AutoTask* task) const override;
  void record_scalar_reductions(AutoTask* task, GlobalRedopID redop) const override;

  void generate_constraints(
    AutoTask* task,
    std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
    const Variable* partition_symbol) const override;

  [[nodiscard]] ArrayAnalyzable to_launcher_arg(
    const std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
    const Strategy& strategy,
    const Domain& launch_domain,
    const std::optional<SymbolicPoint>& projection,
    Legion::PrivilegeMode privilege,
    GlobalRedopID redop) const override;
  [[nodiscard]] ArrayAnalyzable to_launcher_arg_for_fixup(
    const Domain& launch_domain, Legion::PrivilegeMode privilege) const override;

  void collect_storage_trackers(SmallVector<UserStorageTracker>& trackers) const override;
  void calculate_pack_size(TaskReturnLayoutForUnpack* layout) const override;

 private:
  InternalSharedPtr<Type> type_{};
  std::optional<InternalSharedPtr<LogicalStore>> null_mask_{};
  SmallVector<InternalSharedPtr<LogicalArray>> fields_{};
};

}  // namespace legate::detail

#include <legate/data/detail/logical_array.inl>
