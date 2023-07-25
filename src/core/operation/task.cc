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

#include "core/operation/task.h"

#include "core/data/detail/scalar.h"
#include "core/operation/detail/task.h"
#include "core/partitioning/detail/constraint.h"

namespace legate {

////////////////////////////////////////////////////
// legate::AutoTask
////////////////////////////////////////////////////

void AutoTask::add_input(LogicalStore store, Variable partition_symbol)
{
  impl_->add_input(store.impl(), partition_symbol.impl());
}

void AutoTask::add_output(LogicalStore store, Variable partition_symbol)
{
  impl_->add_output(store.impl(), partition_symbol.impl());
}

void AutoTask::add_reduction(LogicalStore store, ReductionOpKind redop, Variable partition_symbol)
{
  impl_->add_reduction(store.impl(), static_cast<int32_t>(redop), partition_symbol.impl());
}

void AutoTask::add_reduction(LogicalStore store, int32_t redop, Variable partition_symbol)
{
  impl_->add_reduction(store.impl(), redop, partition_symbol.impl());
}

void AutoTask::add_scalar_arg(const Scalar& scalar) { impl_->add_scalar_arg(*scalar.impl_); }

void AutoTask::add_scalar_arg(Scalar&& scalar) { impl_->add_scalar_arg(std::move(*scalar.impl_)); }

void AutoTask::add_constraint(Constraint&& constraint)
{
  impl_->add_constraint(std::unique_ptr<detail::Constraint>(constraint.release()));
}

Variable AutoTask::find_or_declare_partition(LogicalStore store)
{
  return Variable(impl_->find_or_declare_partition(store.impl()));
}

Variable AutoTask::declare_partition() { return Variable(impl_->declare_partition()); }

const std::string& AutoTask::provenance() const { return impl_->provenance(); }

void AutoTask::set_concurrent(bool concurrent) { impl_->set_concurrent(concurrent); }

void AutoTask::set_side_effect(bool has_side_effect) { impl_->set_side_effect(has_side_effect); }

void AutoTask::throws_exception(bool can_throw_exception)
{
  impl_->throws_exception(can_throw_exception);
}

void AutoTask::add_communicator(const std::string& name) { impl_->add_communicator(name); }

AutoTask::AutoTask(AutoTask&&) = default;

AutoTask& AutoTask::operator=(AutoTask&&) = default;

AutoTask::~AutoTask() {}

AutoTask::AutoTask(std::unique_ptr<detail::AutoTask> impl) : impl_(std::move(impl)) {}

////////////////////////////////////////////////////
// legate::ManualTask
////////////////////////////////////////////////////

void ManualTask::add_input(LogicalStore store) { impl_->add_input(store.impl()); }

void ManualTask::add_input(LogicalStorePartition store_partition)
{
  impl_->add_input(store_partition.impl());
}

void ManualTask::add_output(LogicalStore store) { impl_->add_output(store.impl()); }

void ManualTask::add_output(LogicalStorePartition store_partition)
{
  impl_->add_output(store_partition.impl());
}

void ManualTask::add_reduction(LogicalStore store, ReductionOpKind redop)
{
  impl_->add_reduction(store.impl(), static_cast<int32_t>(redop));
}

void ManualTask::add_reduction(LogicalStore store, int32_t redop)
{
  impl_->add_reduction(store.impl(), redop);
}

void ManualTask::add_reduction(LogicalStorePartition store_partition, ReductionOpKind redop)
{
  impl_->add_reduction(store_partition.impl(), static_cast<int32_t>(redop));
}

void ManualTask::add_reduction(LogicalStorePartition store_partition, int32_t redop)
{
  impl_->add_reduction(store_partition.impl(), redop);
}

void ManualTask::add_scalar_arg(const Scalar& scalar) { impl_->add_scalar_arg(*scalar.impl_); }

void ManualTask::add_scalar_arg(Scalar&& scalar)
{
  impl_->add_scalar_arg(std::move(*scalar.impl_));
}

const std::string& ManualTask::provenance() const { return impl_->provenance(); }

void ManualTask::set_concurrent(bool concurrent) { impl_->set_concurrent(concurrent); }

void ManualTask::set_side_effect(bool has_side_effect) { impl_->set_side_effect(has_side_effect); }

void ManualTask::throws_exception(bool can_throw_exception)
{
  impl_->throws_exception(can_throw_exception);
}

void ManualTask::add_communicator(const std::string& name) { impl_->add_communicator(name); }

ManualTask::ManualTask(ManualTask&&) = default;

ManualTask& ManualTask::operator=(ManualTask&&) = default;

ManualTask::~ManualTask() {}

ManualTask::ManualTask(std::unique_ptr<detail::ManualTask> impl) : impl_(std::move(impl)) {}

}  // namespace legate
