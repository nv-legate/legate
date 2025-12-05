/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/array_kind.h>
#include <legate/data/detail/logical_array.h>
#include <legate/data/detail/user_storage_tracker.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/operation/projection.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace legate::detail {

class PhysicalArray;
class BasePhysicalArray;
class LogicalStore;
class Shape;

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
  /**
   * @brief Return a broadcasted view to this `BaseLogicalArray`.
   */
  [[nodiscard]] InternalSharedPtr<LogicalArray> broadcast(std::int32_t dim,
                                                          std::size_t dim_size) const override;
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

}  // namespace legate::detail

#include <legate/data/detail/logical_arrays/base_logical_array.inl>
