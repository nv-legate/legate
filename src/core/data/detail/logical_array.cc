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

#include "core/data/detail/physical_array.h"
#include "core/operation/detail/launcher_arg.h"
#include "core/operation/detail/task.h"
#include "core/partitioning/detail/constraint.h"
#include "core/partitioning/detail/constraint_solver.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace legate::detail {

InternalSharedPtr<LogicalStore> LogicalArray::data() const
{
  throw std::invalid_argument{"Data store of a nested array cannot be retrieved"};
  return {};
}

/*static*/ InternalSharedPtr<LogicalArray> LogicalArray::from_store(
  InternalSharedPtr<LogicalStore> store)
{
  return make_internal_shared<BaseLogicalArray>(std::move(store));
}

bool BaseLogicalArray::unbound() const
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(!nullable() || data_->unbound() == null_mask_->unbound());
  }
  return data_->unbound();
}

InternalSharedPtr<LogicalArray> BaseLogicalArray::promote(int32_t extra_dim, size_t dim_size) const
{
  auto null_mask = nullable() ? null_mask_->promote(extra_dim, dim_size) : nullptr;
  auto data      = data_->promote(extra_dim, dim_size);
  return make_internal_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

InternalSharedPtr<LogicalArray> BaseLogicalArray::project(int32_t dim, int64_t index) const
{
  auto null_mask = nullable() ? null_mask_->project(dim, index) : nullptr;
  auto data      = data_->project(dim, index);
  return make_internal_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

InternalSharedPtr<LogicalArray> BaseLogicalArray::slice(int32_t dim, Slice sl) const
{
  auto null_mask = nullable() ? slice_store(null_mask_, dim, sl) : nullptr;
  auto data      = slice_store(data_, dim, sl);
  return make_internal_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

InternalSharedPtr<LogicalArray> BaseLogicalArray::transpose(const std::vector<int32_t>& axes) const
{
  auto null_mask = nullable() ? null_mask_->transpose(axes) : nullptr;
  auto data      = data_->transpose(axes);
  return make_internal_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

InternalSharedPtr<LogicalArray> BaseLogicalArray::delinearize(
  int32_t dim, const std::vector<uint64_t>& sizes) const
{
  auto null_mask = nullable() ? null_mask_->delinearize(dim, sizes) : nullptr;
  auto data      = data_->delinearize(dim, sizes);
  return make_internal_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

InternalSharedPtr<LogicalStore> BaseLogicalArray::null_mask() const
{
  if (!nullable()) {
    throw std::invalid_argument{"Invalid to retrieve the null mask of a non-nullable array"};
  }
  return null_mask_;
}

InternalSharedPtr<PhysicalArray> BaseLogicalArray::get_physical_array() const
{
  return _get_physical_array();
}

InternalSharedPtr<BasePhysicalArray> BaseLogicalArray::_get_physical_array() const
{
  auto data_store = data_->get_physical_store();
  InternalSharedPtr<PhysicalStore> null_mask_store{};
  if (null_mask_ != nullptr) {
    null_mask_store = null_mask_->get_physical_store();
  }
  return make_internal_shared<BasePhysicalArray>(std::move(data_store), std::move(null_mask_store));
}

InternalSharedPtr<LogicalArray> BaseLogicalArray::child(uint32_t /*index*/) const
{
  throw std::invalid_argument{"Non-nested array has no child sub-array"};
  return {};
}

void BaseLogicalArray::record_scalar_or_unbound_outputs(AutoTask* task) const
{
  if (data_->unbound()) {
    task->record_unbound_output(data_);
  } else if (data_->has_scalar_storage()) {
    task->record_scalar_output(data_);
  }

  if (!nullable()) {
    return;
  }

  if (null_mask_->unbound()) {
    task->record_unbound_output(null_mask_);
  } else if (null_mask_->has_scalar_storage()) {
    task->record_scalar_output(null_mask_);
  }
}

void BaseLogicalArray::record_scalar_reductions(AutoTask* task, Legion::ReductionOpID redop) const
{
  if (data_->has_scalar_storage()) {
    task->record_scalar_reduction(data_, redop);
  }
  if (nullable() && null_mask_->has_scalar_storage()) {
    auto null_redop = bool_()->find_reduction_operator(ReductionOpKind::MUL);
    task->record_scalar_reduction(null_mask_, static_cast<Legion::ReductionOpID>(null_redop));
  }
}

void BaseLogicalArray::generate_constraints(
  AutoTask* task,
  std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
  const Variable* partition_symbol) const
{
  mapping.try_emplace(data_, partition_symbol);

  if (!nullable()) {
    return;
  }
  auto part_null_mask = task->declare_partition();
  mapping.try_emplace(null_mask_, part_null_mask);
  task->add_constraint(align(partition_symbol, part_null_mask));
}

std::unique_ptr<Analyzable> BaseLogicalArray::to_launcher_arg(
  const std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
  const Strategy& strategy,
  const Domain& launch_domain,
  const std::optional<SymbolicPoint>& projection,
  Legion::PrivilegeMode privilege,
  int32_t redop) const
{
  auto data_arg = store_to_launcher_arg(
    data_, mapping.at(data_), strategy, launch_domain, projection, privilege, redop);
  std::unique_ptr<Analyzable> null_mask_arg{};

  if (nullable()) {
    auto null_redop =
      privilege == LEGION_REDUCE ? bool_()->find_reduction_operator(ReductionOpKind::MUL) : -1;
    null_mask_arg = store_to_launcher_arg(null_mask_,
                                          mapping.at(null_mask_),
                                          strategy,
                                          launch_domain,
                                          projection,
                                          privilege,
                                          null_redop);
  }

  return std::make_unique<BaseArrayArg>(std::move(data_arg), std::move(null_mask_arg));
}

std::unique_ptr<Analyzable> BaseLogicalArray::to_launcher_arg_for_fixup(
  const Domain& launch_domain, Legion::PrivilegeMode privilege) const
{
  return std::make_unique<BaseArrayArg>(
    store_to_launcher_arg_for_fixup(data_, launch_domain, privilege));
}

bool ListLogicalArray::unbound() const { return descriptor_->unbound() || vardata_->unbound(); }

InternalSharedPtr<LogicalArray> ListLogicalArray::promote(int32_t, size_t) const
{
  throw std::runtime_error{"List array does not support store transformations"};
  return {};
}

InternalSharedPtr<LogicalArray> ListLogicalArray::project(int32_t, int64_t) const
{
  throw std::runtime_error{"List array does not support store transformations"};
  return {};
}

InternalSharedPtr<LogicalArray> ListLogicalArray::slice(int32_t, Slice) const
{
  throw std::runtime_error{"List array does not support store transformations"};
  return {};
}

InternalSharedPtr<LogicalArray> ListLogicalArray::transpose(const std::vector<int32_t>&) const
{
  throw std::runtime_error{"List array does not support store transformations"};
  return {};
}

InternalSharedPtr<LogicalArray> ListLogicalArray::delinearize(int32_t,
                                                              const std::vector<uint64_t>&) const
{
  throw std::runtime_error{"List array does not support store transformations"};
  return {};
}

InternalSharedPtr<PhysicalArray> ListLogicalArray::get_physical_array() const
{
  auto desc_arr    = descriptor_->_get_physical_array();
  auto vardata_arr = vardata_->get_physical_array();
  return make_internal_shared<ListPhysicalArray>(
    type_, std::move(desc_arr), std::move(vardata_arr));
}

InternalSharedPtr<LogicalArray> ListLogicalArray::child(uint32_t index) const
{
  if (unbound()) {
    throw std::invalid_argument{"Invalid to retrieve a sub-array of an unbound array"};
  }
  switch (index) {
    case 0: return descriptor_;
    case 1: return vardata_;
    default: {
      throw std::out_of_range{"List array does not have child " + std::to_string(index)};
      break;
    }
  }
  return nullptr;
}

InternalSharedPtr<BaseLogicalArray> ListLogicalArray::descriptor() const
{
  if (unbound()) {
    throw std::invalid_argument{"Invalid to retrieve a sub-array of an unbound array"};
  }
  return descriptor_;
}

InternalSharedPtr<LogicalArray> ListLogicalArray::vardata() const
{
  if (unbound()) {
    throw std::invalid_argument{"Invalid to retrieve a sub-array of an unbound array"};
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
  std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
  const Variable* partition_symbol) const
{
  descriptor_->generate_constraints(task, mapping, partition_symbol);
  auto part_vardata = task->declare_partition();
  vardata_->generate_constraints(task, mapping, part_vardata);
  if (!unbound()) {
    task->add_constraint(image(partition_symbol, part_vardata));
  }
}

std::unique_ptr<Analyzable> ListLogicalArray::to_launcher_arg(
  const std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
  const Strategy& strategy,
  const Domain& launch_domain,
  const std::optional<SymbolicPoint>& projection,
  Legion::PrivilegeMode privilege,
  int32_t redop) const
{
  const auto desc_priv =
    (LEGION_READ_ONLY == privilege || vardata_->unbound()) ? privilege : LEGION_READ_WRITE;
  auto descriptor_arg =
    descriptor_->to_launcher_arg(mapping, strategy, launch_domain, projection, desc_priv, redop);
  auto vardata_arg =
    vardata_->to_launcher_arg(mapping, strategy, launch_domain, projection, privilege, redop);

  return std::make_unique<ListArrayArg>(type(), std::move(descriptor_arg), std::move(vardata_arg));
}

std::unique_ptr<Analyzable> ListLogicalArray::to_launcher_arg_for_fixup(
  const Domain& launch_domain, Legion::PrivilegeMode privilege) const
{
  auto descriptor_arg = descriptor_->to_launcher_arg_for_fixup(launch_domain, LEGION_READ_WRITE);
  auto vardata_arg    = vardata_->to_launcher_arg_for_fixup(launch_domain, privilege);

  return std::make_unique<ListArrayArg>(type(), std::move(descriptor_arg), std::move(vardata_arg));
}

uint32_t StructLogicalArray::dim() const { return fields_.front()->dim(); }

const InternalSharedPtr<Shape>& StructLogicalArray::shape() const
{
  return fields_.front()->shape();
}

size_t StructLogicalArray::volume() const { return fields_.front()->volume(); }

bool StructLogicalArray::unbound() const
{
  return std::any_of(fields_.begin(), fields_.end(), [](auto& array) { return array->unbound(); });
}

namespace {

// provide a generic template in case the return type of the vector is different from
// fields/needs to be manually specified
template <typename U, typename T, typename F>
std::vector<U> make_array_from_op(const std::vector<T>& fields, F&& generator_fn)
{
  std::vector<U> result;

  result.reserve(fields.size());
  std::transform(
    fields.begin(), fields.end(), std::back_inserter(result), std::forward<F>(generator_fn));
  return result;
}

// in case the return type of the vector is the same as fields
template <typename T, typename F>
std::vector<T> make_array_from_op(const std::vector<T>& fields, F&& generator_fn)
{
  return make_array_from_op<T, T, F>(fields, std::forward<F>(generator_fn));
}

}  // namespace

InternalSharedPtr<LogicalArray> StructLogicalArray::promote(int32_t extra_dim,
                                                            size_t dim_size) const
{
  auto null_mask = nullable() ? null_mask_->promote(extra_dim, dim_size) : nullptr;
  auto fields =
    make_array_from_op(fields_, [&](auto& field) { return field->promote(extra_dim, dim_size); });

  return make_internal_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

InternalSharedPtr<LogicalArray> StructLogicalArray::project(int32_t dim, int64_t index) const
{
  auto null_mask = nullable() ? null_mask_->project(dim, index) : nullptr;
  auto fields =
    make_array_from_op(fields_, [&](auto& field) { return field->project(dim, index); });

  return make_internal_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

InternalSharedPtr<LogicalArray> StructLogicalArray::slice(int32_t dim, Slice sl) const
{
  auto null_mask = nullable() ? slice_store(null_mask_, dim, sl) : nullptr;
  auto fields    = make_array_from_op(fields_, [&](auto& field) { return field->slice(dim, sl); });
  return make_internal_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

InternalSharedPtr<LogicalArray> StructLogicalArray::transpose(
  const std::vector<int32_t>& axes) const
{
  auto null_mask = nullable() ? null_mask_->transpose(axes) : nullptr;
  auto fields    = make_array_from_op(fields_, [&](auto& field) { return field->transpose(axes); });
  return make_internal_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

InternalSharedPtr<LogicalArray> StructLogicalArray::delinearize(
  int32_t dim, const std::vector<uint64_t>& sizes) const
{
  auto null_mask = nullable() ? null_mask_->delinearize(dim, sizes) : nullptr;
  auto fields =
    make_array_from_op(fields_, [&](auto& field) { return field->delinearize(dim, sizes); });
  return make_internal_shared<StructLogicalArray>(type_, std::move(null_mask), std::move(fields));
}

InternalSharedPtr<LogicalStore> StructLogicalArray::null_mask() const
{
  if (!nullable()) {
    throw std::invalid_argument{"Invalid to retrieve the null mask of a non-nullable array"};
  }
  return null_mask_;
}

InternalSharedPtr<PhysicalArray> StructLogicalArray::get_physical_array() const
{
  InternalSharedPtr<PhysicalStore> null_mask_store = nullptr;
  if (null_mask_ != nullptr) {
    null_mask_store = null_mask_->get_physical_store();
  }

  auto field_arrays = make_array_from_op<InternalSharedPtr<PhysicalArray>>(
    fields_, [&](auto& field) { return field->get_physical_array(); });

  return make_internal_shared<StructPhysicalArray>(
    type_, std::move(null_mask_store), std::move(field_arrays));
}

InternalSharedPtr<LogicalArray> StructLogicalArray::child(uint32_t index) const
{
  if (unbound()) {
    throw std::invalid_argument{"Invalid to retrieve a sub-array of an unbound array"};
  }
  return fields_.at(index);
}

InternalSharedPtr<LogicalStore> StructLogicalArray::primary_store() const
{
  return fields_.front()->primary_store();
}

void StructLogicalArray::record_scalar_or_unbound_outputs(AutoTask* task) const
{
  for (auto& field : fields_) {
    field->record_scalar_or_unbound_outputs(task);
  }

  if (!nullable()) {
    return;
  }

  if (null_mask_->unbound()) {
    task->record_unbound_output(null_mask_);
  } else if (null_mask_->has_scalar_storage()) {
    task->record_scalar_output(null_mask_);
  }
}

void StructLogicalArray::record_scalar_reductions(AutoTask* task, Legion::ReductionOpID redop) const
{
  for (auto& field : fields_) {
    field->record_scalar_reductions(task, redop);
  }
  if (nullable() && null_mask_->has_scalar_storage()) {
    auto null_redop = bool_()->find_reduction_operator(ReductionOpKind::MUL);
    task->record_scalar_reduction(null_mask_, static_cast<Legion::ReductionOpID>(null_redop));
  }
}

void StructLogicalArray::generate_constraints(
  AutoTask* task,
  std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
  const Variable* partition_symbol) const
{
  auto it = fields_.begin();
  (*it++)->generate_constraints(task, mapping, partition_symbol);
  for (; it != fields_.end(); ++it) {
    auto part_field = task->declare_partition();
    (*it)->generate_constraints(task, mapping, part_field);
    task->add_constraint(image(partition_symbol, part_field));
  }

  if (!nullable()) {
    return;
  }
  auto part_null_mask = task->declare_partition();
  mapping.try_emplace(null_mask_, part_null_mask);
  task->add_constraint(align(partition_symbol, part_null_mask));
}

std::unique_ptr<Analyzable> StructLogicalArray::to_launcher_arg(
  const std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
  const Strategy& strategy,
  const Domain& launch_domain,
  const std::optional<SymbolicPoint>& projection,
  Legion::PrivilegeMode privilege,
  int32_t redop) const
{
  std::unique_ptr<Analyzable> null_mask_arg = nullptr;
  if (nullable()) {
    auto null_redop =
      privilege == LEGION_REDUCE ? bool_()->find_reduction_operator(ReductionOpKind::MUL) : -1;
    null_mask_arg = store_to_launcher_arg(null_mask_,
                                          mapping.at(null_mask_),
                                          strategy,
                                          launch_domain,
                                          projection,
                                          privilege,
                                          null_redop);
  }

  auto field_args = make_array_from_op<std::unique_ptr<Analyzable>>(fields_, [&](auto& field) {
    return field->to_launcher_arg(mapping, strategy, launch_domain, projection, privilege, redop);
  });

  return std::make_unique<StructArrayArg>(type(), std::move(null_mask_arg), std::move(field_args));
}

std::unique_ptr<Analyzable> StructLogicalArray::to_launcher_arg_for_fixup(
  const Domain& launch_domain, Legion::PrivilegeMode privilege) const
{
  return std::make_unique<StructArrayArg>(
    type(), nullptr, make_array_from_op<std::unique_ptr<Analyzable>>(fields_, [&](auto& field) {
      return field->to_launcher_arg_for_fixup(launch_domain, privilege);
    }));
}

}  // namespace legate::detail
