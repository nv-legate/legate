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

namespace legate::detail {

struct Array;
class AutoTask;
class BaseArray;
struct ConstraintSolver;
class ListLogicalArray;
class Variable;

struct LogicalArray {
  virtual ~LogicalArray() {}
  virtual int32_t dim() const                = 0;
  virtual ArrayKind kind() const             = 0;
  virtual std::shared_ptr<Type> type() const = 0;
  virtual const Shape& extents() const       = 0;
  virtual size_t volume() const              = 0;
  virtual bool unbound() const               = 0;
  virtual bool nullable() const              = 0;
  virtual bool nested() const                = 0;
  virtual uint32_t num_children() const      = 0;

  virtual std::shared_ptr<LogicalArray> promote(int32_t extra_dim, size_t dim_size) const    = 0;
  virtual std::shared_ptr<LogicalArray> project(int32_t dim, int64_t index) const            = 0;
  virtual std::shared_ptr<LogicalArray> slice(int32_t dim, Slice sl) const                   = 0;
  virtual std::shared_ptr<LogicalArray> transpose(const std::vector<int32_t>& axes) const    = 0;
  virtual std::shared_ptr<LogicalArray> delinearize(int32_t dim,
                                                    const std::vector<int64_t>& sizes) const = 0;

  virtual std::shared_ptr<LogicalStore> data() const;
  virtual std::shared_ptr<LogicalStore> null_mask() const           = 0;
  virtual std::shared_ptr<Array> get_physical_array() const         = 0;
  virtual std::shared_ptr<LogicalArray> child(uint32_t index) const = 0;
  virtual std::shared_ptr<LogicalStore> primary_store() const       = 0;

  virtual void record_scalar_or_unbound_outputs(AutoTask* task) const                      = 0;
  virtual void record_scalar_reductions(AutoTask* task, Legion::ReductionOpID redop) const = 0;

  virtual void generate_constraints(
    AutoTask* task,
    std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
    const Variable* partition_symbol) const = 0;

  virtual std::unique_ptr<Analyzable> to_launcher_arg(
    const std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
    const Strategy& strategy,
    const Domain& launch_domain,
    Legion::PrivilegeMode privilege,
    int32_t redop) const = 0;
  virtual std::unique_ptr<Analyzable> to_launcher_arg_for_fixup(
    const Domain& launch_domain, Legion::PrivilegeMode privilege) const = 0;

  [[nodiscard]] static std::shared_ptr<LogicalArray> from_store(
    std::shared_ptr<LogicalStore> store);
};

class BaseLogicalArray : public LogicalArray {
 public:
  BaseLogicalArray(std::shared_ptr<LogicalStore> data,
                   std::shared_ptr<LogicalStore> null_mask = nullptr);

 public:
  int32_t dim() const override { return data_->dim(); }
  ArrayKind kind() const override { return ArrayKind::BASE; }
  std::shared_ptr<Type> type() const override { return data_->type(); }
  const Shape& extents() const override { return data_->extents(); }
  size_t volume() const override { return data_->volume(); }
  bool unbound() const override;
  bool nullable() const override { return null_mask_ != nullptr; }
  bool nested() const override { return false; }
  uint32_t num_children() const override { return 0; }

 public:
  std::shared_ptr<LogicalArray> promote(int32_t extra_dim, size_t dim_size) const override;
  std::shared_ptr<LogicalArray> project(int32_t dim, int64_t index) const override;
  std::shared_ptr<LogicalArray> slice(int32_t dim, Slice sl) const override;
  std::shared_ptr<LogicalArray> transpose(const std::vector<int32_t>& axes) const override;
  std::shared_ptr<LogicalArray> delinearize(int32_t dim,
                                            const std::vector<int64_t>& sizes) const override;

 public:
  std::shared_ptr<LogicalStore> data() const override { return data_; }
  std::shared_ptr<LogicalStore> null_mask() const override;
  std::shared_ptr<Array> get_physical_array() const override;
  std::shared_ptr<BaseArray> _get_physical_array() const;
  std::shared_ptr<LogicalArray> child(uint32_t index) const override;
  std::shared_ptr<LogicalStore> primary_store() const override { return data(); }

 public:
  void record_scalar_or_unbound_outputs(AutoTask* task) const override;
  void record_scalar_reductions(AutoTask* task, Legion::ReductionOpID redop) const override;

 public:
  void generate_constraints(AutoTask* task,
                            std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
                            const Variable* partition_symbol) const override;

 public:
  std::unique_ptr<Analyzable> to_launcher_arg(
    const std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
    const Strategy& strategy,
    const Domain& launch_domain,
    Legion::PrivilegeMode privilege,
    int32_t redop) const override;
  std::unique_ptr<Analyzable> to_launcher_arg_for_fixup(
    const Domain& launch_domain, Legion::PrivilegeMode privilege) const override;

 private:
  std::shared_ptr<LogicalStore> data_;
  std::shared_ptr<LogicalStore> null_mask_;
};

class ListLogicalArray : public LogicalArray {
 public:
  ListLogicalArray(std::shared_ptr<Type> type,
                   std::shared_ptr<BaseLogicalArray> descriptor,
                   std::shared_ptr<LogicalArray> vardata);

 public:
  int32_t dim() const override { return descriptor_->dim(); }
  ArrayKind kind() const override { return ArrayKind::LIST; }
  std::shared_ptr<Type> type() const override { return type_; }
  const Shape& extents() const override { return descriptor_->extents(); }
  size_t volume() const override { return descriptor_->volume(); }
  bool unbound() const override;
  bool nullable() const override { return descriptor_->nullable(); }
  bool nested() const override { return true; }
  uint32_t num_children() const override { return 2; }

 public:
  std::shared_ptr<LogicalArray> promote(int32_t extra_dim, size_t dim_size) const override;
  std::shared_ptr<LogicalArray> project(int32_t dim, int64_t index) const override;
  std::shared_ptr<LogicalArray> slice(int32_t dim, Slice sl) const override;
  std::shared_ptr<LogicalArray> transpose(const std::vector<int32_t>& axes) const override;
  std::shared_ptr<LogicalArray> delinearize(int32_t dim,
                                            const std::vector<int64_t>& sizes) const override;

 public:
  std::shared_ptr<LogicalStore> null_mask() const override { return descriptor_->null_mask(); }
  std::shared_ptr<Array> get_physical_array() const override;
  std::shared_ptr<LogicalArray> child(uint32_t index) const override;
  std::shared_ptr<LogicalStore> primary_store() const override
  {
    return descriptor_->primary_store();
  }
  std::shared_ptr<BaseLogicalArray> descriptor() const;
  std::shared_ptr<LogicalArray> vardata() const;

 public:
  void record_scalar_or_unbound_outputs(AutoTask* task) const override;
  void record_scalar_reductions(AutoTask* task, Legion::ReductionOpID redop) const override;

 public:
  void generate_constraints(AutoTask* task,
                            std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
                            const Variable* partition_symbol) const override;

 public:
  std::unique_ptr<Analyzable> to_launcher_arg(
    const std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
    const Strategy& strategy,
    const Domain& launch_domain,
    Legion::PrivilegeMode privilege,
    int32_t redop) const override;
  std::unique_ptr<Analyzable> to_launcher_arg_for_fixup(
    const Domain& launch_domain, Legion::PrivilegeMode privilege) const override;

 private:
  std::shared_ptr<Type> type_;
  std::shared_ptr<BaseLogicalArray> descriptor_;
  std::shared_ptr<LogicalArray> vardata_;
};

class StructLogicalArray : public LogicalArray {
 public:
  StructLogicalArray(std::shared_ptr<Type> type,
                     std::shared_ptr<LogicalStore> null_mask,
                     std::vector<std::shared_ptr<LogicalArray>>&& fields);

 public:
  int32_t dim() const override;
  ArrayKind kind() const override { return ArrayKind::STRUCT; }
  std::shared_ptr<Type> type() const override { return type_; }
  const Shape& extents() const override;
  size_t volume() const override;
  bool unbound() const override;
  bool nullable() const override { return null_mask_ != nullptr; }
  bool nested() const override { return true; }
  uint32_t num_children() const override { return fields_.size(); }

 public:
  std::shared_ptr<LogicalArray> promote(int32_t extra_dim, size_t dim_size) const override;
  std::shared_ptr<LogicalArray> project(int32_t dim, int64_t index) const override;
  std::shared_ptr<LogicalArray> slice(int32_t dim, Slice sl) const override;
  std::shared_ptr<LogicalArray> transpose(const std::vector<int32_t>& axes) const override;
  std::shared_ptr<LogicalArray> delinearize(int32_t dim,
                                            const std::vector<int64_t>& sizes) const override;

 public:
  std::shared_ptr<LogicalStore> null_mask() const override;
  std::shared_ptr<Array> get_physical_array() const override;
  std::shared_ptr<LogicalArray> child(uint32_t index) const override;
  std::shared_ptr<LogicalStore> primary_store() const override;

 public:
  void record_scalar_or_unbound_outputs(AutoTask* task) const override;
  void record_scalar_reductions(AutoTask* task, Legion::ReductionOpID redop) const override;

 public:
  void generate_constraints(AutoTask* task,
                            std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
                            const Variable* partition_symbol) const override;

 public:
  std::unique_ptr<Analyzable> to_launcher_arg(
    const std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
    const Strategy& strategy,
    const Domain& launch_domain,
    Legion::PrivilegeMode privilege,
    int32_t redop) const override;
  std::unique_ptr<Analyzable> to_launcher_arg_for_fixup(
    const Domain& launch_domain, Legion::PrivilegeMode privilege) const override;

 private:
  std::shared_ptr<Type> type_;
  std::shared_ptr<LogicalStore> null_mask_;
  std::vector<std::shared_ptr<LogicalArray>> fields_;
};

}  // namespace legate::detail
