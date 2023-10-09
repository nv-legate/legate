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

#include "core/data/detail/logical_array.h"
#include "core/data/detail/array.h"
#include "core/operation/detail/launcher_arg.h"
#include "core/operation/detail/task.h"
#include "core/partitioning/detail/constraint.h"
#include "core/partitioning/detail/constraint_solver.h"

namespace legate::detail {

std::shared_ptr<LogicalStore> LogicalArray::data() const
{
  throw std::invalid_argument("Data store of a nested array cannot be retrieved");
  return nullptr;
}

BaseLogicalArray::BaseLogicalArray(std::shared_ptr<LogicalStore> data,
                                   std::shared_ptr<LogicalStore> null_mask)
  : data_(std::move(data)), null_mask_(std::move(null_mask))
{
  assert(data_ != nullptr);
}

bool BaseLogicalArray::unbound() const
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(!nullable() || data_->unbound() == null_mask_->unbound());
  }
  return data_->unbound();
}

std::shared_ptr<LogicalArray> BaseLogicalArray::promote(int32_t extra_dim, size_t dim_size) const
{
  auto null_mask = nullable() ? null_mask_->promote(extra_dim, dim_size) : nullptr;
  auto data      = data_->promote(extra_dim, dim_size);
  return std::make_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

std::shared_ptr<LogicalArray> BaseLogicalArray::project(int32_t dim, int64_t index) const
{
  auto null_mask = nullable() ? null_mask_->project(dim, index) : nullptr;
  auto data      = data_->project(dim, index);
  return std::make_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

std::shared_ptr<LogicalArray> BaseLogicalArray::slice(int32_t dim, Slice sl) const
{
  auto null_mask = nullable() ? null_mask_->slice(dim, sl) : nullptr;
  auto data      = data_->slice(dim, sl);
  return std::make_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

std::shared_ptr<LogicalArray> BaseLogicalArray::transpose(const std::vector<int32_t>& axes) const
{
  auto null_mask = nullable() ? null_mask_->transpose(axes) : nullptr;
  auto data      = data_->transpose(axes);
  return std::make_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

std::shared_ptr<LogicalArray> BaseLogicalArray::delinearize(int32_t dim,
                                                            const std::vector<int64_t>& sizes) const
{
  auto null_mask = nullable() ? null_mask_->delinearize(dim, sizes) : nullptr;
  auto data      = data_->delinearize(dim, sizes);
  return std::make_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

std::shared_ptr<LogicalStore> BaseLogicalArray::null_mask() const
{
  if (!nullable())
    throw std::invalid_argument("Invalid to retrieve the null mask of a non-nullable array");
  return null_mask_;
}

std::shared_ptr<Array> BaseLogicalArray::get_physical_array() const
{
  return _get_physical_array();
}

std::shared_ptr<BaseArray> BaseLogicalArray::_get_physical_array() const
{
  auto data_store                        = data_->get_physical_store();
  std::shared_ptr<Store> null_mask_store = nullptr;
  if (null_mask_ != nullptr) { null_mask_store = null_mask_->get_physical_store(); }
  return std::make_shared<BaseArray>(std::move(data_store), std::move(null_mask_store));
}

std::shared_ptr<LogicalArray> BaseLogicalArray::child(uint32_t index) const
{
  throw std::invalid_argument("Non-nested array has no child sub-array");
  return nullptr;
}

void BaseLogicalArray::record_scalar_or_unbound_outputs(AutoTask* task) const
{
  if (data_->unbound())
    task->record_unbound_output(data_);
  else if (data_->has_scalar_storage())
    task->record_scalar_output(data_);

  if (!nullable()) return;

  if (null_mask_->unbound())
    task->record_unbound_output(null_mask_);
  else if (null_mask_->has_scalar_storage())
    task->record_scalar_output(null_mask_);
}

void BaseLogicalArray::record_scalar_reductions(AutoTask* task, Legion::ReductionOpID redop) const
{
  if (data_->has_scalar_storage()) { task->record_scalar_reduction(data_, redop); }
  if (nullable() && null_mask_->has_scalar_storage()) {
    auto null_redop = bool_()->find_reduction_operator(ReductionOpKind::MUL);
    task->record_scalar_reduction(null_mask_, null_redop);
  }
}

void BaseLogicalArray::generate_constraints(
  AutoTask* task,
  std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
  const Variable* partition_symbol) const
{
  mapping.insert({data_, partition_symbol});

  if (!nullable()) return;
  auto part_null_mask = task->declare_partition();
  mapping.insert({null_mask_, part_null_mask});
  task->add_constraint(align(partition_symbol, part_null_mask));
}

std::unique_ptr<Analyzable> BaseLogicalArray::to_launcher_arg(
  const std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
  const Strategy& strategy,
  const Domain& launch_domain,
  Legion::PrivilegeMode privilege,
  int32_t redop) const
{
  auto data_arg =
    data_->to_launcher_arg(mapping.at(data_), strategy, launch_domain, privilege, redop);
  std::unique_ptr<Analyzable> null_mask_arg = nullptr;
  if (nullable()) {
    auto null_redop =
      privilege == LEGION_REDUCE ? bool_()->find_reduction_operator(ReductionOpKind::MUL) : -1;
    null_mask_arg = null_mask_->to_launcher_arg(
      mapping.at(null_mask_), strategy, launch_domain, privilege, null_redop);
  }

  return std::make_unique<BaseArrayArg>(std::move(data_arg), std::move(null_mask_arg));
}

std::unique_ptr<Analyzable> BaseLogicalArray::to_launcher_arg_for_fixup(
  const Domain& launch_domain, Legion::PrivilegeMode privilege) const
{
  auto data_arg = data_->to_launcher_arg_for_fixup(launch_domain, privilege);
  return std::make_unique<BaseArrayArg>(std::move(data_arg));
}

ListLogicalArray::ListLogicalArray(std::shared_ptr<Type> type,
                                   std::shared_ptr<BaseLogicalArray> descriptor,
                                   std::shared_ptr<LogicalArray> vardata)
  : type_(std::move(type)), descriptor_(std::move(descriptor)), vardata_(std::move(vardata))
{
}

bool ListLogicalArray::unbound() const { return descriptor_->unbound() || vardata_->unbound(); }

std::shared_ptr<LogicalArray> ListLogicalArray::promote(int32_t extra_dim, size_t dim_size) const
{
  throw std::runtime_error("List array does not support store transformations");
  return nullptr;
}

std::shared_ptr<LogicalArray> ListLogicalArray::project(int32_t dim, int64_t index) const
{
  throw std::runtime_error("List array does not support store transformations");
  return nullptr;
}

std::shared_ptr<LogicalArray> ListLogicalArray::slice(int32_t dim, Slice sl) const
{
  throw std::runtime_error("List array does not support store transformations");
  return nullptr;
}

std::shared_ptr<LogicalArray> ListLogicalArray::transpose(const std::vector<int32_t>& axes) const
{
  throw std::runtime_error("List array does not support store transformations");
  return nullptr;
}

std::shared_ptr<LogicalArray> ListLogicalArray::delinearize(int32_t dim,
                                                            const std::vector<int64_t>& sizes) const
{
  throw std::runtime_error("List array does not support store transformations");
  return nullptr;
}

std::shared_ptr<Array> ListLogicalArray::get_physical_array() const
{
  auto desc_arr    = descriptor_->_get_physical_array();
  auto vardata_arr = vardata_->get_physical_array();
  return std::make_shared<ListArray>(type_, std::move(desc_arr), std::move(vardata_arr));
}

std::shared_ptr<LogicalArray> ListLogicalArray::child(uint32_t index) const
{
  if (unbound()) {
    throw std::invalid_argument("Invalid to retrieve a sub-array of an unbound array");
  }
  switch (index) {
    case 0: return descriptor_;
    case 1: return vardata_;
    default: {
      throw std::out_of_range("List array does not have child " + std::to_string(index));
      break;
    }
  }
  return nullptr;
}

std::shared_ptr<BaseLogicalArray> ListLogicalArray::descriptor() const
{
  if (unbound()) {
    throw std::invalid_argument("Invalid to retrieve a sub-array of an unbound array");
  }
  return descriptor_;
}

std::shared_ptr<LogicalArray> ListLogicalArray::vardata() const
{
  if (unbound()) {
    throw std::invalid_argument("Invalid to retrieve a sub-array of an unbound array");
  }
  return vardata_;
}

void ListLogicalArray::record_scalar_or_unbound_outputs(AutoTask* task) const
{
  descriptor_->record_scalar_or_unbound_outputs(task);
  vardata_->record_scalar_or_unbound_outputs(task);
}

void ListLogicalArray::record_scalar_reductions(AutoTask* task, Legion::ReductionOpID redop) const
{
  vardata_->record_scalar_reductions(task, redop);
}

void ListLogicalArray::generate_constraints(
  AutoTask* task,
  std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
  const Variable* partition_symbol) const
{
  descriptor_->generate_constraints(task, mapping, partition_symbol);
  auto part_vardata = task->declare_partition();
  vardata_->generate_constraints(task, mapping, part_vardata);
  if (!unbound()) { task->add_constraint(image(partition_symbol, part_vardata)); }
}

std::unique_ptr<Analyzable> ListLogicalArray::to_launcher_arg(
  const std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
  const Strategy& strategy,
  const Domain& launch_domain,
  Legion::PrivilegeMode privilege,
  int32_t redop) const
{
  auto desc_priv =
    (LEGION_READ_ONLY == privilege || vardata_->unbound()) ? privilege : LEGION_READ_WRITE;
  auto descriptor_arg =
    descriptor_->to_launcher_arg(mapping, strategy, launch_domain, desc_priv, redop);
  auto vardata_arg = vardata_->to_launcher_arg(mapping, strategy, launch_domain, privilege, redop);

  return std::make_unique<ListArrayArg>(type(), std::move(descriptor_arg), std::move(vardata_arg));
}

std::unique_ptr<Analyzable> ListLogicalArray::to_launcher_arg_for_fixup(
  const Domain& launch_domain, Legion::PrivilegeMode privilege) const
{
  auto descriptor_arg = descriptor_->to_launcher_arg_for_fixup(launch_domain, LEGION_READ_WRITE);
  auto vardata_arg    = vardata_->to_launcher_arg_for_fixup(launch_domain, privilege);

  return std::make_unique<ListArrayArg>(type(), std::move(descriptor_arg), std::move(vardata_arg));
}

StructLogicalArray::StructLogicalArray(std::shared_ptr<Type> type,
                                       std::shared_ptr<LogicalStore> null_mask,
                                       std::vector<std::shared_ptr<LogicalArray>>&& fields)
  : type_(std::move(type)), null_mask_(std::move(null_mask)), fields_(std::move(fields))
{
}

int32_t StructLogicalArray::dim() const { return fields_.front()->dim(); }

const Shape& StructLogicalArray::extents() const { return fields_.front()->extents(); }

size_t StructLogicalArray::volume() const { return fields_.front()->volume(); }

bool StructLogicalArray::unbound() const
{
  return std::any_of(fields_.begin(), fields_.end(), [](auto& array) { return array->unbound(); });
}

std::shared_ptr<LogicalArray> StructLogicalArray::promote(int32_t extra_dim, size_t dim_size) const
{
  auto null_mask = nullable() ? null_mask_->promote(extra_dim, dim_size) : nullptr;
  std::vector<std::shared_ptr<LogicalArray>> fields;
  for (auto& field : fields_) { fields.push_back(field->promote(extra_dim, dim_size)); }
  return std::make_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

std::shared_ptr<LogicalArray> StructLogicalArray::project(int32_t dim, int64_t index) const
{
  auto null_mask = nullable() ? null_mask_->project(dim, index) : nullptr;
  std::vector<std::shared_ptr<LogicalArray>> fields;
  for (auto& field : fields_) { fields.push_back(field->project(dim, index)); }
  return std::make_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

std::shared_ptr<LogicalArray> StructLogicalArray::slice(int32_t dim, Slice sl) const
{
  auto null_mask = nullable() ? null_mask_->slice(dim, sl) : nullptr;
  std::vector<std::shared_ptr<LogicalArray>> fields;
  for (auto& field : fields_) { fields.push_back(field->slice(dim, sl)); }
  return std::make_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

std::shared_ptr<LogicalArray> StructLogicalArray::transpose(const std::vector<int32_t>& axes) const
{
  auto null_mask = nullable() ? null_mask_->transpose(axes) : nullptr;
  std::vector<std::shared_ptr<LogicalArray>> fields;
  for (auto& field : fields_) { fields.push_back(field->transpose(axes)); }
  return std::make_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

std::shared_ptr<LogicalArray> StructLogicalArray::delinearize(
  int32_t dim, const std::vector<int64_t>& sizes) const
{
  auto null_mask = nullable() ? null_mask_->delinearize(dim, sizes) : nullptr;
  std::vector<std::shared_ptr<LogicalArray>> fields;
  for (auto& field : fields_) { fields.push_back(field->delinearize(dim, sizes)); }
  return std::make_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

std::shared_ptr<LogicalStore> StructLogicalArray::null_mask() const
{
  if (!nullable())
    throw std::invalid_argument("Invalid to retrieve the null mask of a non-nullable array");
  return null_mask_;
}

std::shared_ptr<Array> StructLogicalArray::get_physical_array() const
{
  std::shared_ptr<Store> null_mask_store = nullptr;
  if (null_mask_ != nullptr) { null_mask_store = null_mask_->get_physical_store(); }
  std::vector<std::shared_ptr<Array>> field_arrays;
  for (auto& field : fields_) { field_arrays.push_back(field->get_physical_array()); }
  return std::make_shared<StructArray>(type_, std::move(null_mask_store), std::move(field_arrays));
}

std::shared_ptr<LogicalArray> StructLogicalArray::child(uint32_t index) const
{
  if (unbound()) {
    throw std::invalid_argument("Invalid to retrieve a sub-array of an unbound array");
  }
  return fields_.at(index);
}

std::shared_ptr<LogicalStore> StructLogicalArray::primary_store() const
{
  return fields_.front()->primary_store();
}

void StructLogicalArray::record_scalar_or_unbound_outputs(AutoTask* task) const
{
  for (auto& field : fields_) { field->record_scalar_or_unbound_outputs(task); }

  if (!nullable()) return;

  if (null_mask_->unbound())
    task->record_unbound_output(null_mask_);
  else if (null_mask_->has_scalar_storage())
    task->record_scalar_output(null_mask_);
}

void StructLogicalArray::record_scalar_reductions(AutoTask* task, Legion::ReductionOpID redop) const
{
  for (auto& field : fields_) { field->record_scalar_reductions(task, redop); }
  if (nullable() && null_mask_->has_scalar_storage()) {
    auto null_redop = bool_()->find_reduction_operator(ReductionOpKind::MUL);
    task->record_scalar_reduction(null_mask_, null_redop);
  }
}

void StructLogicalArray::generate_constraints(
  AutoTask* task,
  std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
  const Variable* partition_symbol) const
{
  auto it = fields_.begin();
  (*it++)->generate_constraints(task, mapping, partition_symbol);
  for (; it != fields_.end(); ++it) {
    auto part_field = task->declare_partition();
    (*it)->generate_constraints(task, mapping, part_field);
    task->add_constraint(image(partition_symbol, part_field));
  }
}

std::unique_ptr<Analyzable> StructLogicalArray::to_launcher_arg(
  const std::map<std::shared_ptr<LogicalStore>, const Variable*>& mapping,
  const Strategy& strategy,
  const Domain& launch_domain,
  Legion::PrivilegeMode privilege,
  int32_t redop) const
{
  std::unique_ptr<Analyzable> null_mask_arg = nullptr;
  if (nullable()) {
    auto null_redop =
      privilege == LEGION_REDUCE ? bool_()->find_reduction_operator(ReductionOpKind::MUL) : -1;
    null_mask_arg = null_mask_->to_launcher_arg(
      mapping.at(null_mask_), strategy, launch_domain, privilege, null_redop);
  }

  std::vector<std::unique_ptr<Analyzable>> field_args;
  for (auto& field : fields_) {
    field_args.push_back(
      field->to_launcher_arg(mapping, strategy, launch_domain, privilege, redop));
  }

  return std::make_unique<StructArrayArg>(type(), std::move(null_mask_arg), std::move(field_args));
}

std::unique_ptr<Analyzable> StructLogicalArray::to_launcher_arg_for_fixup(
  const Domain& launch_domain, Legion::PrivilegeMode privilege) const
{
  std::vector<std::unique_ptr<Analyzable>> field_args;
  for (auto& field : fields_) {
    field_args.push_back(field->to_launcher_arg_for_fixup(launch_domain, privilege));
  }
  return std::make_unique<StructArrayArg>(type(), nullptr, std::move(field_args));
}

}  // namespace legate::detail
