/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/operation/task.h>

#include <legate/operation/detail/task.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

namespace legate {

////////////////////////////////////////////////////
// legate::AutoTask
////////////////////////////////////////////////////

class AutoTask::Impl {
 public:
  explicit Impl(InternalSharedPtr<detail::AutoTask> impl) : impl_{std::move(impl)} {}

  [[nodiscard]] const SharedPtr<detail::LogicalArray>& add_ref(LogicalArray array)
  {
    auto&& inserted = refs_.emplace_back(std::move(array));
    return inserted.impl();
  }

  void clear_refs() { refs_.clear(); }

  [[nodiscard]] const SharedPtr<detail::AutoTask>& impl() const noexcept { return impl_; }
  [[nodiscard]] SharedPtr<detail::AutoTask>& impl() noexcept { return impl_; }

 private:
  SharedPtr<detail::AutoTask> impl_{};
  std::vector<LogicalArray> refs_{};
};

// ==========================================================================================

const SharedPtr<detail::AutoTask>& AutoTask::impl_() const
{
  auto&& result = pimpl_->impl();
  if (!result) {
    throw detail::TracedException<std::runtime_error>{
      "Illegal to reuse task descriptors that are already submitted"};
  }
  return result;
}

SharedPtr<detail::AutoTask> AutoTask::release_()
{
  auto&& result = std::move(pimpl_->impl());
  return result;
}

InternalSharedPtr<detail::LogicalArray> AutoTask::record_user_ref_(LogicalArray array)
{
  return pimpl_->add_ref(std::move(array));
}

void AutoTask::clear_user_refs_() { pimpl_->clear_refs(); }

// ==========================================================================================

Variable AutoTask::add_input(LogicalArray array)
{
  return Variable{impl_()->add_input(record_user_ref_(std::move(array)))};
}

Variable AutoTask::add_output(LogicalArray array)
{
  return Variable{impl_()->add_output(record_user_ref_(std::move(array)))};
}

Variable AutoTask::add_reduction(LogicalArray array, ReductionOpKind redop_kind)
{
  return add_reduction(std::move(array), static_cast<std::int32_t>(redop_kind));
}

Variable AutoTask::add_reduction(LogicalArray array, std::int32_t redop_kind)
{
  return Variable{impl_()->add_reduction(record_user_ref_(std::move(array)), redop_kind)};
}

Variable AutoTask::add_input(LogicalArray array, Variable partition_symbol)
{
  impl_()->add_input(record_user_ref_(std::move(array)), partition_symbol.impl());
  return partition_symbol;
}

Variable AutoTask::add_output(LogicalArray array, Variable partition_symbol)
{
  impl_()->add_output(record_user_ref_(std::move(array)), partition_symbol.impl());
  return partition_symbol;
}

Variable AutoTask::add_reduction(LogicalArray array,
                                 ReductionOpKind redop_kind,
                                 Variable partition_symbol)
{
  return add_reduction(
    std::move(array), static_cast<std::int32_t>(redop_kind), std::move(partition_symbol));
}

Variable AutoTask::add_reduction(LogicalArray array,
                                 std::int32_t redop_kind,
                                 Variable partition_symbol)
{
  impl_()->add_reduction(record_user_ref_(std::move(array)), redop_kind, partition_symbol.impl());
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

std::string_view AutoTask::provenance() const { return impl_()->provenance().as_string_view(); }

void AutoTask::set_concurrent(bool concurrent) { impl_()->set_concurrent(concurrent); }

void AutoTask::set_side_effect(bool has_side_effect) { impl_()->set_side_effect(has_side_effect); }

void AutoTask::throws_exception(bool can_throw_exception)
{
  impl_()->throws_exception(can_throw_exception);
}

void AutoTask::add_communicator(std::string_view name) { impl_()->add_communicator(name); }

AutoTask::AutoTask(InternalSharedPtr<detail::AutoTask> impl)
  : pimpl_{make_internal_shared<Impl>(std::move(impl))}
{
}

AutoTask::~AutoTask() noexcept = default;

////////////////////////////////////////////////////
// legate::ManualTask
////////////////////////////////////////////////////

class ManualTask::Impl {
 public:
  explicit Impl(InternalSharedPtr<detail::ManualTask> impl) : impl_{std::move(impl)} {}

  [[nodiscard]] const SharedPtr<detail::LogicalStore>& add_ref(LogicalStore store)
  {
    return store_refs_.emplace_back(std::move(store)).impl();
  }
  [[nodiscard]] const SharedPtr<detail::LogicalStorePartition>& add_ref(
    LogicalStorePartition store_partition)
  {
    return part_refs_.emplace_back(std::move(store_partition)).impl();
  }

  void clear_refs()
  {
    store_refs_.clear();
    part_refs_.clear();
  }

  [[nodiscard]] const SharedPtr<detail::ManualTask>& impl() const noexcept { return impl_; }
  [[nodiscard]] SharedPtr<detail::ManualTask>& impl() noexcept { return impl_; }

 private:
  SharedPtr<detail::ManualTask> impl_{};
  std::vector<LogicalStore> store_refs_{};
  std::vector<LogicalStorePartition> part_refs_{};
};

// ==========================================================================================

const SharedPtr<detail::ManualTask>& ManualTask::impl_() const
{
  auto&& result = pimpl_->impl();
  if (!result) {
    throw detail::TracedException<std::runtime_error>{
      "Illegal to reuse task descriptors that are already submitted"};
  }
  return result;
}

SharedPtr<detail::ManualTask> ManualTask::release_()
{
  auto&& result = std::move(pimpl_->impl());
  return result;
}

InternalSharedPtr<detail::LogicalStore> ManualTask::record_user_ref_(LogicalStore store)
{
  return pimpl_->add_ref(std::move(store));
}

InternalSharedPtr<detail::LogicalStorePartition> ManualTask::record_user_ref_(
  LogicalStorePartition store_partition)
{
  return pimpl_->add_ref(std::move(store_partition));
}

void ManualTask::clear_user_refs_() { pimpl_->clear_refs(); }

// ==========================================================================================

void ManualTask::add_input(LogicalStore store)
{
  impl_()->add_input(record_user_ref_(std::move(store)));
}

void ManualTask::add_input(LogicalStorePartition store_partition,
                           std::optional<SymbolicPoint> projection)
{
  impl_()->add_input(record_user_ref_(std::move(store_partition)), std::move(projection));
}

void ManualTask::add_output(LogicalStore store)
{
  impl_()->add_output(record_user_ref_(std::move(store)));
}

void ManualTask::add_output(LogicalStorePartition store_partition,
                            std::optional<SymbolicPoint> projection)
{
  impl_()->add_output(record_user_ref_(std::move(store_partition)), std::move(projection));
}

void ManualTask::add_reduction(LogicalStore store, ReductionOpKind redop_kind)
{
  add_reduction(std::move(store), static_cast<std::int32_t>(redop_kind));
}

void ManualTask::add_reduction(LogicalStore store, std::int32_t redop_kind)
{
  impl_()->add_reduction(record_user_ref_(std::move(store)), redop_kind);
}

void ManualTask::add_reduction(LogicalStorePartition store_partition,
                               ReductionOpKind redop_kind,
                               std::optional<SymbolicPoint> projection)
{
  add_reduction(
    std::move(store_partition), static_cast<std::int32_t>(redop_kind), std::move(projection));
}

void ManualTask::add_reduction(LogicalStorePartition store_partition,
                               std::int32_t redop_kind,
                               std::optional<SymbolicPoint> projection)
{
  impl_()->add_reduction(
    record_user_ref_(std::move(store_partition)), redop_kind, std::move(projection));
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

void ManualTask::add_communicator(std::string_view name) { impl_()->add_communicator(name); }

ManualTask::ManualTask(InternalSharedPtr<detail::ManualTask> impl)
  : pimpl_{make_internal_shared<Impl>(std::move(impl))}
{
}

ManualTask::~ManualTask() noexcept = default;

}  // namespace legate
