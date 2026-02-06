/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/task.h>

#include <legate/data/physical_array.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/operation/detail/task.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>
#include <variant>

namespace legate {

namespace {

[[nodiscard]] mapping::StoreTarget get_inline_store_target()
{
  const auto processor = detail::Runtime::get_runtime().get_executing_processor();
  const auto variant   = mapping::detail::to_variant_code(processor.kind());
  switch (variant) {
    case VariantCode::CPU: return mapping::StoreTarget::SYSMEM;
    case VariantCode::GPU: return mapping::StoreTarget::FBMEM;
    case VariantCode::OMP: return mapping::StoreTarget::SOCKETMEM;
  }
  LEGATE_ABORT("Unhandled variant code");
}

}  // namespace

////////////////////////////////////////////////////
// legate::AutoTask
////////////////////////////////////////////////////

class AutoTask::Impl {
 public:
  using TaskVariant = std::variant<SharedPtr<detail::AutoTask>, SharedPtr<detail::PhysicalTask>>;

  explicit Impl(InternalSharedPtr<detail::AutoTask> impl)
    : task_{SharedPtr<detail::AutoTask>{std::move(impl)}}
  {
  }

  explicit Impl(InternalSharedPtr<detail::PhysicalTask> impl)
    : task_{SharedPtr<detail::PhysicalTask>{std::move(impl)}}
  {
  }

  [[nodiscard]] const SharedPtr<detail::LogicalArray>& add_ref(LogicalArray array)
  {
    auto&& inserted = refs_.emplace_back(std::move(array));
    return inserted.impl();
  }

  void clear_refs() { refs_.clear(); }

  [[nodiscard]] bool is_physical_task() const noexcept
  {
    return std::holds_alternative<SharedPtr<detail::PhysicalTask>>(task_);
  }

  [[nodiscard]] const SharedPtr<detail::AutoTask>& auto_task_impl() const
  {
    auto* ptr = std::get_if<SharedPtr<detail::AutoTask>>(&task_);
    if (!ptr || !*ptr) {
      throw detail::TracedException<std::runtime_error>{
        "Illegal to reuse task descriptors that are already submitted"};
    }
    return *ptr;
  }

  [[nodiscard]] SharedPtr<detail::AutoTask>& auto_task_impl()
  {
    auto* ptr = std::get_if<SharedPtr<detail::AutoTask>>(&task_);
    if (!ptr) {
      throw detail::TracedException<std::runtime_error>{
        "Cannot access AutoTask impl from PhysicalTask variant"};
    }
    return *ptr;
  }

  [[nodiscard]] const SharedPtr<detail::PhysicalTask>& physical_task_impl() const
  {
    auto* ptr = std::get_if<SharedPtr<detail::PhysicalTask>>(&task_);
    if (!ptr || !*ptr) {
      throw detail::TracedException<std::runtime_error>{
        "Cannot access PhysicalTask impl from AutoTask variant"};
    }
    return *ptr;
  }

  [[nodiscard]] SharedPtr<detail::PhysicalTask>& physical_task_impl()
  {
    auto* ptr = std::get_if<SharedPtr<detail::PhysicalTask>>(&task_);
    if (!ptr) {
      throw detail::TracedException<std::runtime_error>{
        "Cannot access PhysicalTask impl from AutoTask variant"};
    }
    return *ptr;
  }

  [[nodiscard]] SharedPtr<detail::AutoTask> release_auto_task()
  {
    return std::move(auto_task_impl());
  }

  [[nodiscard]] SharedPtr<detail::PhysicalTask> release_physical_task()
  {
    return std::move(physical_task_impl());
  }

 private:
  TaskVariant task_{};
  std::vector<LogicalArray> refs_{};
};

// ==========================================================================================

const SharedPtr<detail::AutoTask>& AutoTask::impl_() const { return pimpl_->auto_task_impl(); }

SharedPtr<detail::AutoTask> AutoTask::release_() { return pimpl_->release_auto_task(); }

SharedPtr<detail::PhysicalTask> AutoTask::release_physical_()
{
  return pimpl_->release_physical_task();
}

bool AutoTask::is_inline_execution_() const { return pimpl_->is_physical_task(); }

InternalSharedPtr<detail::LogicalArray> AutoTask::record_user_ref_(LogicalArray array)
{
  return pimpl_->add_ref(std::move(array));
}

void AutoTask::clear_user_refs_() { pimpl_->clear_refs(); }

// ==========================================================================================

Variable AutoTask::add_input(LogicalArray array)
{
  if (is_inline_execution_()) {
    auto physical_array = array.get_physical_array(get_inline_store_target());
    pimpl_->physical_task_impl()->add_input(physical_array.impl());
    return Variable{nullptr};
  }
  return Variable{impl_()->add_input(record_user_ref_(std::move(array)))};
}

Variable AutoTask::add_output(LogicalArray array)
{
  if (is_inline_execution_()) {
    auto physical_array = array.get_physical_array(get_inline_store_target());
    pimpl_->physical_task_impl()->add_output(physical_array.impl());
    return Variable{nullptr};
  }
  return Variable{impl_()->add_output(record_user_ref_(std::move(array)))};
}

Variable AutoTask::add_reduction(LogicalArray array, ReductionOpKind redop_kind)
{
  return add_reduction(std::move(array), static_cast<std::int32_t>(redop_kind));
}

Variable AutoTask::add_reduction(LogicalArray array, std::int32_t redop_kind)
{
  if (is_inline_execution_()) {
    auto physical_array = array.get_physical_array(get_inline_store_target());
    pimpl_->physical_task_impl()->add_reduction(physical_array.impl(), redop_kind);
    return Variable{nullptr};
  }
  return Variable{impl_()->add_reduction(record_user_ref_(std::move(array)), redop_kind)};
}

Variable AutoTask::add_input(LogicalArray array, Variable partition_symbol)
{
  if (is_inline_execution_()) {
    auto physical_array = array.get_physical_array(get_inline_store_target());
    pimpl_->physical_task_impl()->add_input(physical_array.impl());
    return Variable{nullptr};
  }
  impl_()->add_input(record_user_ref_(std::move(array)), partition_symbol.impl());
  return partition_symbol;
}

Variable AutoTask::add_output(LogicalArray array, Variable partition_symbol)
{
  if (is_inline_execution_()) {
    auto physical_array = array.get_physical_array(get_inline_store_target());
    pimpl_->physical_task_impl()->add_output(physical_array.impl());
    return Variable{nullptr};
  }
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
  if (is_inline_execution_()) {
    auto physical_array = array.get_physical_array(get_inline_store_target());
    pimpl_->physical_task_impl()->add_reduction(physical_array.impl(), redop_kind);
    return Variable{nullptr};
  }
  impl_()->add_reduction(record_user_ref_(std::move(array)), redop_kind, partition_symbol.impl());
  return partition_symbol;
}

void AutoTask::add_scalar_arg(  // NOLINT(readability-make-member-function-const)
  const Scalar& scalar)
{
  if (is_inline_execution_()) {
    pimpl_->physical_task_impl()->add_scalar_arg(scalar.impl());
    return;
  }
  impl_()->add_scalar_arg(scalar.impl());
}

void AutoTask::add_constraint(  // NOLINT(readability-make-member-function-const)
  const Constraint& constraint)
{
  if (is_inline_execution_()) {
    return;  // No-op for inline execution (single partition)
  }
  impl_()->add_constraint(constraint.impl());
}

void AutoTask::add_constraints(Span<const Constraint> constraints)
{
  for (auto&& c : constraints) {
    add_constraint(c);
  }
}

Variable AutoTask::find_or_declare_partition(  // NOLINT(readability-make-member-function-const)
  const LogicalArray& array)
{
  if (is_inline_execution_()) {
    throw detail::TracedException<std::runtime_error>{
      "Partitioning is not supported for inline task execution"};
  }
  return Variable{impl_()->find_or_declare_partition(array.impl())};
}

Variable AutoTask::declare_partition()  // NOLINT(readability-make-member-function-const)
{
  if (is_inline_execution_()) {
    throw detail::TracedException<std::runtime_error>{
      "Partitioning is not supported for inline task execution"};
  }
  return Variable{impl_()->declare_partition()};
}

std::string_view AutoTask::provenance() const
{
  if (is_inline_execution_()) {
    return pimpl_->physical_task_impl()->provenance().as_string_view();
  }
  return impl_()->provenance().as_string_view();
}

void AutoTask::set_concurrent(bool concurrent)  // NOLINT(readability-make-member-function-const)
{
  if (is_inline_execution_()) {
    pimpl_->physical_task_impl()->set_concurrent(concurrent);
    return;
  }
  impl_()->set_concurrent(concurrent);
}

void AutoTask::set_side_effect(  // NOLINT(readability-make-member-function-const)
  bool has_side_effect)
{
  if (is_inline_execution_()) {
    pimpl_->physical_task_impl()->set_side_effect(has_side_effect);
    return;
  }
  impl_()->set_side_effect(has_side_effect);
}

void AutoTask::throws_exception(  // NOLINT(readability-make-member-function-const)
  bool can_throw_exception)
{
  if (is_inline_execution_()) {
    pimpl_->physical_task_impl()->throws_exception(can_throw_exception);
    return;
  }
  impl_()->throws_exception(can_throw_exception);
}

void AutoTask::add_communicator(  // NOLINT(readability-make-member-function-const)
  std::string_view name)
{
  if (is_inline_execution_()) {
    throw detail::TracedException<std::runtime_error>{
      "Communicators are not supported for inline task execution"};
  }
  impl_()->add_communicator(name);
}

AutoTask::AutoTask(InternalSharedPtr<detail::AutoTask> impl)
  : pimpl_{make_internal_shared<Impl>(std::move(impl))}
{
}

AutoTask::AutoTask(InternalSharedPtr<detail::PhysicalTask> impl)
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

////////////////////////////////////////////////////
// legate::PhysicalTask
////////////////////////////////////////////////////

PhysicalTask::PhysicalTask(const Key&, InternalSharedPtr<detail::PhysicalTask> impl)
  : pimpl_{std::move(impl)}
{
}

PhysicalTask::~PhysicalTask() noexcept = default;

const SharedPtr<detail::PhysicalTask>& PhysicalTask::impl_() const { return pimpl_; }

SharedPtr<detail::PhysicalTask> PhysicalTask::release_(const Key&) { return std::move(pimpl_); }

void PhysicalTask::add_input(const PhysicalArray& array) const { impl_()->add_input(array.impl()); }

void PhysicalTask::add_output(const PhysicalArray& array) const
{
  impl_()->add_output(array.impl());
}

void PhysicalTask::add_reduction(const PhysicalArray& array, std::int32_t redop_kind) const
{
  impl_()->add_reduction(array.impl(), redop_kind);
}

void PhysicalTask::add_scalar_arg(const Scalar& scalar) const
{
  impl_()->add_scalar_arg(scalar.impl());
}

void PhysicalTask::set_concurrent(bool concurrent) const { impl_()->set_concurrent(concurrent); }

void PhysicalTask::set_side_effect(bool has_side_effect) const
{
  impl_()->set_side_effect(has_side_effect);
}

void PhysicalTask::throws_exception(bool can_throw_exception) const
{
  impl_()->throws_exception(can_throw_exception);
}

}  // namespace legate
