/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/task.h>

#include <legate/data/physical_store.h>
#include <legate/operation/detail/task.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/runtime/runtime.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

namespace legate {

////////////////////////////////////////////////////
// legate::AutoTask
////////////////////////////////////////////////////

const InternalSharedPtr<detail::AutoTask>& AutoTask::impl_() const { return pimpl_; }

Variable AutoTask::add_input(LogicalStore store)
{
  return Variable{impl_()->add_input(std::move(store).impl())};
}

Variable AutoTask::add_output(LogicalStore store)
{
  return Variable{impl_()->add_output(std::move(store).impl())};
}

Variable AutoTask::add_reduction(LogicalStore store,
                                 ReductionOpKind redop_kind,
                                 bool store_has_identity)
{
  return add_reduction(std::move(store), static_cast<std::int32_t>(redop_kind), store_has_identity);
}

Variable AutoTask::add_reduction(LogicalStore store,
                                 std::int32_t redop_kind,
                                 bool store_has_identity)
{
  return Variable{impl_()->add_reduction(std::move(store).impl(), redop_kind, store_has_identity)};
}

Variable AutoTask::add_input(LogicalStore store, Variable partition_symbol)
{
  impl_()->add_input(std::move(store).impl(), partition_symbol.impl());
  return partition_symbol;
}

Variable AutoTask::add_output(LogicalStore store, Variable partition_symbol)
{
  impl_()->add_output(std::move(store).impl(), partition_symbol.impl());
  return partition_symbol;
}

Variable AutoTask::add_reduction(LogicalStore store,
                                 ReductionOpKind redop_kind,
                                 Variable partition_symbol,
                                 bool store_has_identity)
{
  return add_reduction(std::move(store),
                       static_cast<std::int32_t>(redop_kind),
                       std::move(partition_symbol),
                       store_has_identity);
}

Variable AutoTask::add_reduction(LogicalStore store,
                                 std::int32_t redop_kind,
                                 Variable partition_symbol,
                                 bool store_has_identity)
{
  impl_()->add_reduction(
    std::move(store).impl(), redop_kind, partition_symbol.impl(), store_has_identity);
  return partition_symbol;
}

void AutoTask::add_scalar_arg(  // NOLINT(readability-make-member-function-const)
  const Scalar& scalar)
{
  impl_()->add_scalar_arg(scalar.impl());
}

void AutoTask::add_constraint(  // NOLINT(readability-make-member-function-const)
  const Constraint& constraint)
{
  impl_()->add_constraint(constraint.impl(), /*bypass_signature_check=*/false);
}

void AutoTask::add_constraints(Span<const Constraint> constraints)
{
  for (auto&& c : constraints) {
    add_constraint(c);
  }
}

Variable AutoTask::find_or_declare_partition(  // NOLINT(readability-make-member-function-const)
  const LogicalStore& store)
{
  return Variable{impl_()->find_or_declare_partition(store.impl())};
}

Variable AutoTask::declare_partition()  // NOLINT(readability-make-member-function-const)
{
  return Variable{impl_()->declare_partition()};
}

std::string_view AutoTask::provenance() const { return impl_()->provenance().as_string_view(); }

void AutoTask::set_concurrent(bool concurrent)  // NOLINT(readability-make-member-function-const)
{
  impl_()->set_concurrent(concurrent);
}

void AutoTask::set_side_effect(  // NOLINT(readability-make-member-function-const)
  bool has_side_effect)
{
  impl_()->set_side_effect(has_side_effect);
}

void AutoTask::throws_exception(  // NOLINT(readability-make-member-function-const)
  bool can_throw_exception)
{
  impl_()->throws_exception(can_throw_exception);
}

void AutoTask::add_communicator(  // NOLINT(readability-make-member-function-const)
  std::string_view name)
{
  impl_()->add_communicator(name, /*bypass_signature_check=*/false);
}

AutoTask::AutoTask(InternalSharedPtr<detail::AutoTask> impl) : pimpl_{std::move(impl)} {}

AutoTask::~AutoTask() noexcept = default;

////////////////////////////////////////////////////
// legate::ManualTask
////////////////////////////////////////////////////

const InternalSharedPtr<detail::ManualTask>& ManualTask::impl_() const { return pimpl_; }

void ManualTask::add_input(LogicalStore store)
{
  static_cast<void>(impl_()->add_input(std::move(store).impl()));
}

void ManualTask::add_input(LogicalStorePartition store_partition,
                           std::optional<SymbolicPoint> projection,
                           bool is_key_partition)
{
  impl_()->add_input(std::move(store_partition).impl(), std::move(projection), is_key_partition);
}

void ManualTask::add_output(LogicalStore store) { impl_()->add_output(std::move(store).impl()); }

void ManualTask::add_output(LogicalStorePartition store_partition,
                            std::optional<SymbolicPoint> projection,
                            bool is_key_partition)
{
  impl_()->add_output(std::move(store_partition).impl(), std::move(projection), is_key_partition);
}

void ManualTask::add_reduction(LogicalStore store,
                               ReductionOpKind redop_kind,
                               bool store_has_identity)
{
  add_reduction(std::move(store), static_cast<std::int32_t>(redop_kind), store_has_identity);
}

void ManualTask::add_reduction(LogicalStore store, std::int32_t redop_kind, bool store_has_identity)
{
  impl_()->add_reduction(std::move(store).impl(), redop_kind, store_has_identity);
}

void ManualTask::add_reduction(LogicalStorePartition store_partition,
                               ReductionOpKind redop_kind,
                               std::optional<SymbolicPoint> projection,
                               bool is_key_partition,
                               bool store_has_identity)
{
  add_reduction(std::move(store_partition),
                static_cast<std::int32_t>(redop_kind),
                std::move(projection),
                is_key_partition,
                store_has_identity);
}

void ManualTask::add_reduction(LogicalStorePartition store_partition,
                               std::int32_t redop_kind,
                               std::optional<SymbolicPoint> projection,
                               bool is_key_partition,
                               bool store_has_identity)
{
  impl_()->add_reduction(std::move(store_partition).impl(),
                         redop_kind,
                         std::move(projection),
                         is_key_partition,
                         store_has_identity);
}

void ManualTask::add_scalar_arg(const Scalar& scalar) { impl_()->add_scalar_arg(scalar.impl()); }

std::string_view ManualTask::provenance() const { return impl_()->provenance().as_string_view(); }

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
  impl_()->add_communicator(name, /*bypass_signature_check=*/false);
}

ManualTask::ManualTask(InternalSharedPtr<detail::ManualTask> impl) : pimpl_{std::move(impl)} {}

ManualTask::~ManualTask() noexcept = default;

}  // namespace legate
