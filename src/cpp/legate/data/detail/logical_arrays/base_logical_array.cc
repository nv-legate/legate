/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_arrays/base_logical_array.h>

#include <legate/data/detail/physical_arrays/base_physical_array.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/operation/detail/task.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

namespace legate::detail {

bool BaseLogicalArray::unbound() const
{
  LEGATE_ASSERT(!nullable() || data()->unbound() == null_mask()->unbound());
  return data()->unbound();
}

InternalSharedPtr<LogicalArray> BaseLogicalArray::promote(std::int32_t extra_dim,
                                                          std::size_t dim_size) const
{
  auto null_mask =
    nullable() ? std::make_optional(this->null_mask()->promote(extra_dim, dim_size)) : std::nullopt;
  auto data = this->data()->promote(extra_dim, dim_size);
  return make_internal_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

InternalSharedPtr<LogicalArray> BaseLogicalArray::project(std::int32_t dim,
                                                          std::int64_t index) const
{
  auto null_mask =
    nullable() ? std::make_optional(this->null_mask()->project(dim, index)) : std::nullopt;
  auto data = this->data()->project(dim, index);
  return make_internal_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

InternalSharedPtr<LogicalArray> BaseLogicalArray::broadcast(std::int32_t dim,
                                                            std::size_t dim_size) const
{
  auto null_mask =
    nullable() ? std::make_optional(this->null_mask()->broadcast(dim, dim_size)) : std::nullopt;
  auto data = this->data()->broadcast(dim, dim_size);
  return make_internal_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

InternalSharedPtr<LogicalArray> BaseLogicalArray::slice(std::int32_t dim, Slice sl) const
{
  auto null_mask =
    nullable() ? std::make_optional(slice_store(this->null_mask(), dim, sl)) : std::nullopt;
  auto data = slice_store(this->data(), dim, sl);
  return make_internal_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

InternalSharedPtr<LogicalArray> BaseLogicalArray::transpose(
  SmallVector<std::int32_t, LEGATE_MAX_DIM> axes) const
{
  auto null_mask =
    nullable() ? std::make_optional(this->null_mask()->transpose(axes)) : std::nullopt;
  auto data = this->data()->transpose(std::move(axes));
  return make_internal_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

InternalSharedPtr<LogicalArray> BaseLogicalArray::delinearize(
  std::int32_t dim, SmallVector<std::uint64_t, LEGATE_MAX_DIM> sizes) const
{
  auto null_mask =
    nullable() ? std::make_optional(this->null_mask()->delinearize(dim, sizes)) : std::nullopt;
  auto data = this->data()->delinearize(dim, std::move(sizes));
  return make_internal_shared<BaseLogicalArray>(std::move(data), std::move(null_mask));
}

const InternalSharedPtr<LogicalStore>& BaseLogicalArray::null_mask() const
{
  if (!nullable()) {
    throw TracedException<std::invalid_argument>{
      "Invalid to retrieve the null mask of a non-nullable array"};
  }
  return *null_mask_;  // NOLINT(bugprone-unchecked-optional-access)
}

InternalSharedPtr<PhysicalArray> BaseLogicalArray::get_physical_array(
  legate::mapping::StoreTarget target, bool ignore_future_mutability) const
{
  return get_base_physical_array(target, ignore_future_mutability);
}

InternalSharedPtr<BasePhysicalArray> BaseLogicalArray::get_base_physical_array(
  legate::mapping::StoreTarget target, bool ignore_future_mutability) const
{
  auto data_store = data()->get_physical_store(target, ignore_future_mutability);
  std::optional<InternalSharedPtr<PhysicalStore>> null_mask_store{};

  if (nullable()) {
    null_mask_store = null_mask()->get_physical_store(target, ignore_future_mutability);
  }
  return make_internal_shared<BasePhysicalArray>(std::move(data_store), std::move(null_mask_store));
}

InternalSharedPtr<LogicalArray> BaseLogicalArray::child(std::uint32_t /*index*/) const
{
  throw TracedException<std::invalid_argument>{"Non-nested array has no child sub-array"};
}

void BaseLogicalArray::record_scalar_or_unbound_outputs(AutoTask* task) const
{
  if (data()->has_scalar_storage()) {
    task->record_scalar_output(data());
  } else if (data()->unbound()) {
    task->record_unbound_output(data());
  }
  if (!nullable()) {
    return;
  }

  if (null_mask()->has_scalar_storage()) {
    task->record_scalar_output(null_mask());
  } else if (null_mask()->unbound()) {
    task->record_unbound_output(null_mask());
  }
}

void BaseLogicalArray::record_scalar_reductions(AutoTask* task, GlobalRedopID redop) const
{
  if (data()->has_scalar_storage()) {
    task->record_scalar_reduction(data(), redop);
  }
  if (nullable() && null_mask()->has_scalar_storage()) {
    auto null_redop = bool_()->find_reduction_operator(ReductionOpKind::MUL);
    task->record_scalar_reduction(null_mask(), null_redop);
  }
}

void BaseLogicalArray::generate_constraints(
  AutoTask* task,
  std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
  const Variable* partition_symbol) const
{
  mapping.try_emplace(data(), partition_symbol);

  if (!nullable()) {
    return;
  }

  auto [it, inserted] = mapping.try_emplace(null_mask());

  if (inserted) {
    it->second = task->declare_partition();
  }
  // Need to bypass the signature check here because these generated constraints are not
  // technically visible to the user (you cannot declare different constraints on the "main"
  // store and the nullable store in the signature).
  task->add_constraint(align(partition_symbol, it->second), /* bypass_signature_check */ true);
}

ArrayAnalyzable BaseLogicalArray::to_launcher_arg(
  const std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>& mapping,
  const Strategy& strategy,
  const Domain& launch_domain,
  const std::optional<SymbolicPoint>& projection,
  Legion::PrivilegeMode privilege,
  GlobalRedopID redop) const
{
  auto data_arg = store_to_launcher_arg(
    data(), mapping.at(data()), strategy, launch_domain, projection, privilege, redop);
  std::optional<StoreAnalyzable> null_mask_arg{};

  if (nullable()) {
    auto null_redop = privilege == LEGION_REDUCE
                        ? bool_()->find_reduction_operator(ReductionOpKind::MUL)
                        : GlobalRedopID{-1};
    null_mask_arg   = store_to_launcher_arg(null_mask(),
                                          mapping.at(null_mask()),
                                          strategy,
                                          launch_domain,
                                          projection,
                                          privilege,
                                          null_redop);
  }

  return BaseArrayArg{std::move(data_arg), std::move(null_mask_arg)};
}

ArrayAnalyzable BaseLogicalArray::to_launcher_arg_for_fixup(const Domain& launch_domain,
                                                            Legion::PrivilegeMode privilege) const
{
  return BaseArrayArg{store_to_launcher_arg_for_fixup(data(), launch_domain, privilege)};
}

void BaseLogicalArray::collect_storage_trackers(SmallVector<UserStorageTracker>& trackers) const
{
  trackers.emplace_back(data());
  if (nullable()) {
    trackers.emplace_back(null_mask());
  }
}

void BaseLogicalArray::calculate_pack_size(TaskReturnLayoutForUnpack* layout) const
{
  data()->calculate_pack_size(layout);
  if (nullable()) {
    null_mask()->calculate_pack_size(layout);
  }
}

}  // namespace legate::detail
