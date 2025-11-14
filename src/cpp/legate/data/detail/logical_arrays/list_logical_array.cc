/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_arrays/list_logical_array.h>

#include <legate/data/detail/logical_arrays/base_logical_array.h>
#include <legate/data/detail/physical_array.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/operation/detail/task.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/partitioning/detail/constraint_solver.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <stdexcept>

namespace legate::detail {

bool ListLogicalArray::unbound() const { return descriptor_->unbound() || vardata_->unbound(); }

InternalSharedPtr<LogicalArray> ListLogicalArray::promote(std::int32_t, std::size_t) const
{
  throw TracedException<std::runtime_error>{"List array does not support store transformations"};
}

InternalSharedPtr<LogicalArray> ListLogicalArray::project(std::int32_t, std::int64_t) const
{
  throw TracedException<std::runtime_error>{"List array does not support store transformations"};
}

InternalSharedPtr<LogicalArray> ListLogicalArray::broadcast(std::int32_t, std::size_t) const
{
  throw TracedException<std::runtime_error>{"List array does not support store transformations"};
}

InternalSharedPtr<LogicalArray> ListLogicalArray::slice(std::int32_t, Slice) const
{
  throw TracedException<std::runtime_error>{"List array does not support store transformations"};
}

InternalSharedPtr<LogicalArray> ListLogicalArray::transpose(
  SmallVector<std::int32_t, LEGATE_MAX_DIM>) const
{
  throw TracedException<std::runtime_error>{"List array does not support store transformations"};
}

InternalSharedPtr<LogicalArray> ListLogicalArray::delinearize(
  std::int32_t, SmallVector<std::uint64_t, LEGATE_MAX_DIM>) const
{
  throw TracedException<std::runtime_error>{"List array does not support store transformations"};
}

InternalSharedPtr<PhysicalArray> ListLogicalArray::get_physical_array(
  legate::mapping::StoreTarget target, bool ignore_future_mutability) const
{
  auto desc_arr    = descriptor_->get_base_physical_array(target, ignore_future_mutability);
  auto vardata_arr = vardata_->get_physical_array(target, ignore_future_mutability);
  return make_internal_shared<ListPhysicalArray>(
    type_, std::move(desc_arr), std::move(vardata_arr));
}

InternalSharedPtr<LogicalArray> ListLogicalArray::child(std::uint32_t index) const
{
  if (unbound()) {
    throw TracedException<std::invalid_argument>{
      "Invalid to retrieve a sub-array of an unbound array"};
  }
  switch (index) {
    case 0: return descriptor_;
    case 1: return vardata_;
    default: {  // legate-lint: no-switch-default
      throw TracedException<std::out_of_range>{
        fmt::format("List array does not have child {}", index)};
    }
  }
  return nullptr;
}

const InternalSharedPtr<BaseLogicalArray>& ListLogicalArray::descriptor() const
{
  if (unbound()) {
    throw TracedException<std::invalid_argument>{
      "Invalid to retrieve a sub-array of an unbound array"};
  }
  return descriptor_;
}

const InternalSharedPtr<LogicalArray>& ListLogicalArray::vardata() const
{
  if (unbound()) {
    throw TracedException<std::invalid_argument>{
      "Invalid to retrieve a sub-array of an unbound array"};
  }
  return vardata_;
}

void ListLogicalArray::record_scalar_or_unbound_outputs(AutoTask* task) const
{
  descriptor_->record_scalar_or_unbound_outputs(task);
  vardata_->record_scalar_or_unbound_outputs(task);
}

void ListLogicalArray::record_scalar_reductions(AutoTask* task, GlobalRedopID redop) const
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
    // Need to bypass the signature check here because these generated constraints are not
    // technically visible to the user (you cannot declare different constraints on the "main"
    // store and the nullable store in the signature).
    task->add_constraint(image(partition_symbol, part_vardata, ImageComputationHint::FIRST_LAST),
                         /* bypass_signature_check */ true);
  }
}

ArrayAnalyzable ListLogicalArray::to_launcher_arg(
  const std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
  const Strategy& strategy,
  const Domain& launch_domain,
  const std::optional<SymbolicPoint>& projection,
  Legion::PrivilegeMode privilege,
  GlobalRedopID redop) const
{
  const auto desc_priv =
    (LEGION_READ_ONLY == privilege || vardata_->unbound()) ? privilege : LEGION_READ_WRITE;
  auto descriptor_arg =
    descriptor_->to_launcher_arg(mapping, strategy, launch_domain, projection, desc_priv, redop);
  auto vardata_arg =
    vardata_->to_launcher_arg(mapping, strategy, launch_domain, projection, privilege, redop);

  return ListArrayArg{type(), std::move(descriptor_arg), std::move(vardata_arg)};
}

ArrayAnalyzable ListLogicalArray::to_launcher_arg_for_fixup(const Domain& launch_domain,
                                                            Legion::PrivilegeMode privilege) const
{
  auto descriptor_arg = descriptor_->to_launcher_arg_for_fixup(launch_domain, LEGION_READ_WRITE);
  auto vardata_arg    = vardata_->to_launcher_arg_for_fixup(launch_domain, privilege);

  return ListArrayArg{type(), std::move(descriptor_arg), std::move(vardata_arg)};
}

void ListLogicalArray::collect_storage_trackers(SmallVector<UserStorageTracker>& trackers) const
{
  descriptor_->collect_storage_trackers(trackers);
  vardata_->collect_storage_trackers(trackers);
}

void ListLogicalArray::calculate_pack_size(TaskReturnLayoutForUnpack* layout) const
{
  descriptor_->calculate_pack_size(layout);
  vardata_->calculate_pack_size(layout);
}

}  // namespace legate::detail
