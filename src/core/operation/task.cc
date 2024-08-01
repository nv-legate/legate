/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/operation/task.h"

#include "core/data/detail/scalar.h"
#include "core/operation/detail/task.h"
#include "core/partitioning/detail/constraint.h"

namespace legate {

////////////////////////////////////////////////////
// legate::AutoTask
////////////////////////////////////////////////////

[[nodiscard]] const SharedPtr<detail::AutoTask>& AutoTask::impl_() const { return pimpl_; }

// ==========================================================================================

Variable AutoTask::add_input(const LogicalArray& array)
{
  return Variable{impl_()->add_input(array.impl())};
}

Variable AutoTask::add_output(const LogicalArray& array)
{
  return Variable{impl_()->add_output(array.impl())};
}

Variable AutoTask::add_reduction(const LogicalArray& array, ReductionOpKind redop_kind)
{
  return add_reduction(array, static_cast<std::int32_t>(redop_kind));
}

Variable AutoTask::add_reduction(const LogicalArray& array, std::int32_t redop_kind)
{
  return Variable{impl_()->add_reduction(array.impl(), redop_kind)};
}

Variable AutoTask::add_input(const LogicalArray& array, Variable partition_symbol)
{
  impl_()->add_input(array.impl(), partition_symbol.impl());
  return partition_symbol;
}

Variable AutoTask::add_output(const LogicalArray& array, Variable partition_symbol)
{
  impl_()->add_output(array.impl(), partition_symbol.impl());
  return partition_symbol;
}

Variable AutoTask::add_reduction(const LogicalArray& array,
                                 ReductionOpKind redop_kind,
                                 Variable partition_symbol)
{
  return add_reduction(array, static_cast<std::int32_t>(redop_kind), std::move(partition_symbol));
}

Variable AutoTask::add_reduction(const LogicalArray& array,
                                 std::int32_t redop_kind,
                                 Variable partition_symbol)
{
  impl_()->add_reduction(array.impl(), redop_kind, partition_symbol.impl());
  return partition_symbol;
}

void AutoTask::add_scalar_arg(const Scalar& scalar) { impl_()->add_scalar_arg(scalar.impl()); }

void AutoTask::add_constraint(const Constraint& constraint)
{
  impl_()->add_constraint(constraint.impl());
}

Variable AutoTask::find_or_declare_partition(const LogicalArray& array)
{
  return Variable{impl_()->find_or_declare_partition(array.impl())};
}

Variable AutoTask::declare_partition() { return Variable{impl_()->declare_partition()}; }

std::string_view AutoTask::provenance() const { return impl_()->provenance(); }

void AutoTask::set_concurrent(bool concurrent) { impl_()->set_concurrent(concurrent); }

void AutoTask::set_side_effect(bool has_side_effect) { impl_()->set_side_effect(has_side_effect); }

void AutoTask::throws_exception(bool can_throw_exception)
{
  impl_()->throws_exception(can_throw_exception);
}

void AutoTask::add_communicator(std::string_view name)
{
  impl_()->add_communicator(std::move(name));
}

AutoTask::AutoTask(InternalSharedPtr<detail::AutoTask> impl) : pimpl_{std::move(impl)} {}

AutoTask::~AutoTask() noexcept = default;

////////////////////////////////////////////////////
// legate::ManualTask
////////////////////////////////////////////////////

[[nodiscard]] const SharedPtr<detail::ManualTask>& ManualTask::impl_() const { return pimpl_; }

// ==========================================================================================

void ManualTask::add_input(const LogicalStore& store) { impl_()->add_input(store.impl()); }

void ManualTask::add_input(const LogicalStorePartition& store_partition,
                           std::optional<SymbolicPoint> projection)
{
  impl_()->add_input(store_partition.impl(), std::move(projection));
}

void ManualTask::add_output(const LogicalStore& store) { impl_()->add_output(store.impl()); }

void ManualTask::add_output(const LogicalStorePartition& store_partition,
                            std::optional<SymbolicPoint> projection)
{
  impl_()->add_output(store_partition.impl(), std::move(projection));
}

void ManualTask::add_reduction(const LogicalStore& store, ReductionOpKind redop_kind)
{
  add_reduction(store, static_cast<std::int32_t>(redop_kind));
}

void ManualTask::add_reduction(const LogicalStore& store, std::int32_t redop_kind)
{
  impl_()->add_reduction(store.impl(), redop_kind);
}

void ManualTask::add_reduction(const LogicalStorePartition& store_partition,
                               ReductionOpKind redop_kind,
                               std::optional<SymbolicPoint> projection)
{
  add_reduction(store_partition, static_cast<std::int32_t>(redop_kind), std::move(projection));
}

void ManualTask::add_reduction(const LogicalStorePartition& store_partition,
                               std::int32_t redop_kind,
                               std::optional<SymbolicPoint> projection)
{
  impl_()->add_reduction(store_partition.impl(), redop_kind, std::move(projection));
}

void ManualTask::add_scalar_arg(const Scalar& scalar) { impl_()->add_scalar_arg(scalar.impl()); }

std::string_view ManualTask::provenance() const { return impl_()->provenance(); }

void ManualTask::set_concurrent(bool concurrent) { impl_()->set_concurrent(concurrent); }

void ManualTask::set_side_effect(bool has_side_effect)
{
  impl_()->set_side_effect(has_side_effect);
}

void ManualTask::throws_exception(bool can_throw_exception)
{
  impl_()->throws_exception(can_throw_exception);
}

void ManualTask::add_communicator(std::string_view name)
{
  impl_()->add_communicator(std::move(name));
}

ManualTask::ManualTask(InternalSharedPtr<detail::ManualTask> impl) : pimpl_{std::move(impl)} {}

ManualTask::~ManualTask() noexcept = default;

}  // namespace legate
