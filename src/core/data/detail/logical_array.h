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

#include "core/data/detail/array_kind.h"
#include "core/data/detail/logical_store.h"
#include "core/operation/detail/launcher_arg.h"
#include "core/operation/projection.h"

#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace legate::detail {

struct PhysicalArray;
class AutoTask;
class BaseArray;
struct ConstraintSolver;
class ListLogicalArray;
class Variable;

struct LogicalArray {
  LogicalArray(const LogicalArray&)                        = default;
  LogicalArray(LogicalArray&&)                             = delete;
  LogicalArray& operator=(const LogicalArray&)             = default;
  LogicalArray& operator=(LogicalArray&&)                  = delete;
  LogicalArray()                                           = default;
  virtual ~LogicalArray()                                  = default;
  [[nodiscard]] virtual int32_t dim() const                = 0;
  [[nodiscard]] virtual ArrayKind kind() const             = 0;
  [[nodiscard]] virtual std::shared_ptr<Type> type() const = 0;
  [[nodiscard]] virtual const Shape& extents() const       = 0;
  [[nodiscard]] virtual size_t volume() const              = 0;
  [[nodiscard]] virtual bool unbound() const               = 0;
  [[nodiscard]] virtual bool nullable() const              = 0;
  [[nodiscard]] virtual bool nested() const                = 0;
  [[nodiscard]] virtual uint32_t num_children() const      = 0;

  [[nodiscard]] virtual std::shared_ptr<LogicalArray> promote(int32_t extra_dim,
                                                              size_t dim_size) const            = 0;
  [[nodiscard]] virtual std::shared_ptr<LogicalArray> project(int32_t dim, int64_t index) const = 0;
  [[nodiscard]] virtual std::shared_ptr<LogicalArray> slice(int32_t dim, Slice sl) const        = 0;
  [[nodiscard]] virtual std::shared_ptr<LogicalArray> transpose(
    const std::vector<int32_t>& axes) const = 0;
  [[nodiscard]] virtual std::shared_ptr<LogicalArray> delinearize(
    int32_t dim, const std::vector<uint64_t>& sizes) const = 0;

  [[nodiscard]] virtual std::shared_ptr<LogicalStore> data() const;
  [[nodiscard]] virtual std::shared_ptr<LogicalStore> null_mask() const           = 0;
  [[nodiscard]] virtual std::shared_ptr<PhysicalArray> get_physical_array() const = 0;
  [[nodiscard]] virtual std::shared_ptr<LogicalArray> child(uint32_t index) const = 0;
  [[nodiscard]] virtual std::shared_ptr<LogicalStore> primary_store() const       = 0;

  virtual void record_scalar_or_unbound_outputs(AutoTask* task) const                      = 0;
  virtual void record_scalar_reductions(AutoTask* task, Legion::ReductionOpID redop) const = 0;

  virtual void generate_constraints(
    AutoTask* task,
    std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
    const Variable* partition_symbol) const = 0;

  [[nodiscard]] virtual std::unique_ptr<Analyzable> to_launcher_arg(
    const std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
    const Strategy& strategy,
    const Domain& launch_domain,
    const std::optional<SymbolicPoint>& projection,
    Legion::PrivilegeMode privilege,
    int32_t redop) const = 0;
  [[nodiscard]] virtual std::unique_ptr<Analyzable> to_launcher_arg_for_fixup(
    const Domain& launch_domain, Legion::PrivilegeMode privilege) const = 0;

  [[nodiscard]] static std::shared_ptr<LogicalArray> from_store(
    std::shared_ptr<LogicalStore> store);
};

class BaseLogicalArray final : public LogicalArray {
 public:
  BaseLogicalArray(std::shared_ptr<LogicalStore> data,
                   std::shared_ptr<LogicalStore> null_mask = nullptr);

  [[nodiscard]] int32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
  [[nodiscard]] std::shared_ptr<Type> type() const override;
  [[nodiscard]] const Shape& extents() const override;
  [[nodiscard]] size_t volume() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] uint32_t num_children() const override;

  [[nodiscard]] std::shared_ptr<LogicalArray> promote(int32_t extra_dim,
                                                      size_t dim_size) const override;
  [[nodiscard]] std::shared_ptr<LogicalArray> project(int32_t dim, int64_t index) const override;
  [[nodiscard]] std::shared_ptr<LogicalArray> slice(int32_t dim, Slice sl) const override;
  [[nodiscard]] std::shared_ptr<LogicalArray> transpose(
    const std::vector<int32_t>& axes) const override;
  [[nodiscard]] std::shared_ptr<LogicalArray> delinearize(
    int32_t dim, const std::vector<uint64_t>& sizes) const override;

  [[nodiscard]] std::shared_ptr<LogicalStore> data() const override;
  [[nodiscard]] std::shared_ptr<LogicalStore> null_mask() const override;
  [[nodiscard]] std::shared_ptr<PhysicalArray> get_physical_array() const override;
  [[nodiscard]] std::shared_ptr<BasePhysicalArray> _get_physical_array() const;
  [[nodiscard]] std::shared_ptr<LogicalArray> child(uint32_t index) const override;
  [[nodiscard]] std::shared_ptr<LogicalStore> primary_store() const override;

  void record_scalar_or_unbound_outputs(AutoTask* task) const override;
  void record_scalar_reductions(AutoTask* task, Legion::ReductionOpID redop) const override;

  void generate_constraints(AutoTask* task,
                            std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
                            const Variable* partition_symbol) const override;

  [[nodiscard]] std::unique_ptr<Analyzable> to_launcher_arg(
    const std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
    const Strategy& strategy,
    const Domain& launch_domain,
    const std::optional<SymbolicPoint>& projection,
    Legion::PrivilegeMode privilege,
    int32_t redop) const override;
  [[nodiscard]] std::unique_ptr<Analyzable> to_launcher_arg_for_fixup(
    const Domain& launch_domain, Legion::PrivilegeMode privilege) const override;

 private:
  std::shared_ptr<LogicalStore> data_{};
  std::shared_ptr<LogicalStore> null_mask_{};
};

class ListLogicalArray final : public LogicalArray {
 public:
  ListLogicalArray(std::shared_ptr<Type> type,
                   std::shared_ptr<BaseLogicalArray> descriptor,
                   std::shared_ptr<LogicalArray> vardata);

  [[nodiscard]] int32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
  [[nodiscard]] std::shared_ptr<Type> type() const override;
  [[nodiscard]] const Shape& extents() const override;
  [[nodiscard]] size_t volume() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] uint32_t num_children() const override;

  [[nodiscard]] std::shared_ptr<LogicalArray> promote(int32_t extra_dim,
                                                      size_t dim_size) const override;
  [[nodiscard]] std::shared_ptr<LogicalArray> project(int32_t dim, int64_t index) const override;
  [[nodiscard]] std::shared_ptr<LogicalArray> slice(int32_t dim, Slice sl) const override;
  [[nodiscard]] std::shared_ptr<LogicalArray> transpose(
    const std::vector<int32_t>& axes) const override;
  [[nodiscard]] std::shared_ptr<LogicalArray> delinearize(
    int32_t dim, const std::vector<uint64_t>& sizes) const override;

  [[nodiscard]] std::shared_ptr<LogicalStore> null_mask() const override;
  [[nodiscard]] std::shared_ptr<PhysicalArray> get_physical_array() const override;
  [[nodiscard]] std::shared_ptr<LogicalArray> child(uint32_t index) const override;
  [[nodiscard]] std::shared_ptr<LogicalStore> primary_store() const override;
  [[nodiscard]] std::shared_ptr<BaseLogicalArray> descriptor() const;
  [[nodiscard]] std::shared_ptr<LogicalArray> vardata() const;

  void record_scalar_or_unbound_outputs(AutoTask* task) const override;
  void record_scalar_reductions(AutoTask* task, Legion::ReductionOpID redop) const override;

  void generate_constraints(AutoTask* task,
                            std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
                            const Variable* partition_symbol) const override;

  [[nodiscard]] std::unique_ptr<Analyzable> to_launcher_arg(
    const std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
    const Strategy& strategy,
    const Domain& launch_domain,
    const std::optional<SymbolicPoint>& projection,
    Legion::PrivilegeMode privilege,
    int32_t redop) const override;
  [[nodiscard]] std::unique_ptr<Analyzable> to_launcher_arg_for_fixup(
    const Domain& launch_domain, Legion::PrivilegeMode privilege) const override;

 private:
  std::shared_ptr<Type> type_{};
  std::shared_ptr<BaseLogicalArray> descriptor_{};
  std::shared_ptr<LogicalArray> vardata_{};
};

class StructLogicalArray final : public LogicalArray {
 public:
  StructLogicalArray(std::shared_ptr<Type> type,
                     std::shared_ptr<LogicalStore> null_mask,
                     std::vector<std::shared_ptr<LogicalArray>>&& fields);

  [[nodiscard]] int32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
  [[nodiscard]] std::shared_ptr<Type> type() const override;
  [[nodiscard]] const Shape& extents() const override;
  [[nodiscard]] size_t volume() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] uint32_t num_children() const override;

  [[nodiscard]] std::shared_ptr<LogicalArray> promote(int32_t extra_dim,
                                                      size_t dim_size) const override;
  [[nodiscard]] std::shared_ptr<LogicalArray> project(int32_t dim, int64_t index) const override;
  [[nodiscard]] std::shared_ptr<LogicalArray> slice(int32_t dim, Slice sl) const override;
  [[nodiscard]] std::shared_ptr<LogicalArray> transpose(
    const std::vector<int32_t>& axes) const override;
  [[nodiscard]] std::shared_ptr<LogicalArray> delinearize(
    int32_t dim, const std::vector<uint64_t>& sizes) const override;

  [[nodiscard]] std::shared_ptr<LogicalStore> null_mask() const override;
  [[nodiscard]] std::shared_ptr<PhysicalArray> get_physical_array() const override;
  [[nodiscard]] std::shared_ptr<LogicalArray> child(uint32_t index) const override;
  [[nodiscard]] std::shared_ptr<LogicalStore> primary_store() const override;

  void record_scalar_or_unbound_outputs(AutoTask* task) const override;
  void record_scalar_reductions(AutoTask* task, Legion::ReductionOpID redop) const override;

  void generate_constraints(AutoTask* task,
                            std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
                            const Variable* partition_symbol) const override;

  [[nodiscard]] std::unique_ptr<Analyzable> to_launcher_arg(
    const std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
    const Strategy& strategy,
    const Domain& launch_domain,
    const std::optional<SymbolicPoint>& projection,
    Legion::PrivilegeMode privilege,
    int32_t redop) const override;
  [[nodiscard]] std::unique_ptr<Analyzable> to_launcher_arg_for_fixup(
    const Domain& launch_domain, Legion::PrivilegeMode privilege) const override;

 private:
  std::shared_ptr<Type> type_{};
  std::shared_ptr<LogicalStore> null_mask_{};
  std::vector<std::shared_ptr<LogicalArray>> fields_{};
};

}  // namespace legate::detail

#include "core/data/detail/logical_array.inl"
