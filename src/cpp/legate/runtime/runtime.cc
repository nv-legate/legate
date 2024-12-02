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

#include "legate/runtime/runtime.h"

#include "legate/data/detail/shape.h"
#include "legate/operation/detail/task.h"
#include "legate/runtime/detail/runtime.h"
#include "legate/runtime/scope.h"
#include "legate/utilities/detail/tuple.h"
#include "legate/utilities/internal_shared_ptr.h"
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/core.h>
#include <optional>
#include <stdexcept>

namespace legate {

Library Runtime::find_library(std::string_view library_name) const
{
  return Library{impl_->find_library(std::move(library_name), false)};
}

std::optional<Library> Runtime::maybe_find_library(std::string_view library_name) const
{
  if (auto result = impl_->find_library(std::move(library_name), true)) {
    return {Library{std::move(result)}};
  }
  return std::nullopt;
}

Library Runtime::create_library(std::string_view library_name,
                                const ResourceConfig& config,
                                std::unique_ptr<mapping::Mapper> mapper,
                                std::map<VariantCode, VariantOptions> default_options)
{
  return Library{impl_->create_library(
    std::move(library_name), config, std::move(mapper), std::move(default_options))};
}

Library Runtime::find_or_create_library(
  std::string_view library_name,
  const ResourceConfig& config,
  std::unique_ptr<mapping::Mapper> mapper,
  const std::map<VariantCode, VariantOptions>& default_options,
  bool* created)
{
  return Library{impl_->find_or_create_library(
    std::move(library_name), config, std::move(mapper), default_options, created)};
}

AutoTask Runtime::create_task(Library library, LocalTaskID task_id)
{
  return AutoTask{impl_->create_task(library.impl(), task_id)};
}

ManualTask Runtime::create_task(Library library,
                                LocalTaskID task_id,
                                const tuple<std::uint64_t>& launch_shape)
{
  return create_task(library, task_id, detail::to_domain(launch_shape));
}

ManualTask Runtime::create_task(Library library, LocalTaskID task_id, const Domain& launch_domain)
{
  return ManualTask{impl_->create_task(library.impl(), task_id, launch_domain)};
}

void Runtime::issue_copy(LogicalStore& target,
                         const LogicalStore& source,
                         std::optional<ReductionOpKind> redop_kind)
{
  auto op =
    redop_kind ? std::make_optional(static_cast<std::int32_t>(redop_kind.value())) : std::nullopt;
  issue_copy(target, source, op);
}

void Runtime::issue_copy(LogicalStore& target,
                         const LogicalStore& source,
                         std::optional<std::int32_t> redop_kind)
{
  impl_->issue_copy(target.impl(), source.impl(), redop_kind);
}

void Runtime::issue_gather(LogicalStore& target,
                           const LogicalStore& source,
                           const LogicalStore& source_indirect,
                           std::optional<ReductionOpKind> redop_kind)
{
  auto op =
    redop_kind ? std::make_optional(static_cast<std::int32_t>(redop_kind.value())) : std::nullopt;
  issue_gather(target, source, source_indirect, op);
}

void Runtime::issue_gather(LogicalStore& target,
                           const LogicalStore& source,
                           const LogicalStore& source_indirect,
                           std::optional<std::int32_t> redop_kind)
{
  impl_->issue_gather(target.impl(), source.impl(), source_indirect.impl(), redop_kind);
}

void Runtime::issue_scatter(LogicalStore& target,
                            const LogicalStore& target_indirect,
                            const LogicalStore& source,
                            std::optional<ReductionOpKind> redop_kind)
{
  auto op =
    redop_kind ? std::make_optional(static_cast<std::int32_t>(redop_kind.value())) : std::nullopt;
  issue_scatter(target, target_indirect, source, op);
}

void Runtime::issue_scatter(LogicalStore& target,
                            const LogicalStore& target_indirect,
                            const LogicalStore& source,
                            std::optional<std::int32_t> redop_kind)
{
  impl_->issue_scatter(target.impl(), target_indirect.impl(), source.impl(), redop_kind);
}

void Runtime::issue_scatter_gather(LogicalStore& target,
                                   const LogicalStore& target_indirect,
                                   const LogicalStore& source,
                                   const LogicalStore& source_indirect,
                                   std::optional<ReductionOpKind> redop_kind)
{
  auto op =
    redop_kind ? std::make_optional(static_cast<std::int32_t>(redop_kind.value())) : std::nullopt;
  issue_scatter_gather(target, target_indirect, source, source_indirect, op);
}

void Runtime::issue_scatter_gather(LogicalStore& target,
                                   const LogicalStore& target_indirect,
                                   const LogicalStore& source,
                                   const LogicalStore& source_indirect,
                                   std::optional<std::int32_t> redop_kind)
{
  impl_->issue_scatter_gather(
    target.impl(), target_indirect.impl(), source.impl(), source_indirect.impl(), redop_kind);
}

void Runtime::issue_fill(const LogicalArray& lhs, const LogicalStore& value)
{
  impl_->issue_fill(lhs.impl(), value.impl());
}

void Runtime::issue_fill(const LogicalArray& lhs, const Scalar& value)
{
  impl_->issue_fill(lhs.impl(), *value.impl());
}

LogicalStore Runtime::tree_reduce(Library library,
                                  LocalTaskID task_id,
                                  const LogicalStore& store,
                                  std::int32_t radix)
{
  auto out_store = create_store(store.type(), 1);

  impl_->tree_reduce(library.impl(), task_id, store.impl(), out_store.impl(), radix);
  return out_store;
}

void Runtime::submit(AutoTask&& task)
{
  impl_->submit(task.release_());
  // The following line should never be moved before the `submit()` call above, nor should it be
  // fused with the `release_()` call, because their side effects must be ordered properly in the
  // task stream. Logical arrays read by a task must stay alive even after they lose all user
  // references until the task is submitted. Otherwise, their backing region fields will be
  // initialized by discard operations as part of their deallocation even before the reader task
  // runs, leading to an uninitialized access error.
  task.clear_user_refs_();
}

void Runtime::submit(ManualTask&& task)
{
  impl_->submit(task.release_());
  // The following line should never be moved before the `submit()` call above, nor should it be
  // fused with the `release_()` call, because their side effects must be ordered properly in the
  // task stream. Logical arrays read by a task must stay alive even after they lose all user
  // references until the task is submitted. Otherwise, their backing region fields will be
  // initialized by discard operations as part of their deallocation even before the reader task
  // runs, leading to an uninitialized access error.
  task.clear_user_refs_();
}

LogicalArray Runtime::create_array(const Type& type, std::uint32_t dim, bool nullable)
{
  return LogicalArray{impl_->create_array(
    make_internal_shared<detail::Shape>(dim), type.impl(), nullable, false /*optimize_scalar*/)};
}

LogicalArray Runtime::create_array(const Shape& shape,
                                   const Type& type,
                                   bool nullable,
                                   bool optimize_scalar)
{
  auto shape_impl = shape.impl();
  // We shouldn't allow users to create unbound arrays out of the same shape that hasn't be bound
  // yet, because they may get initialized by different producer tasks and there's no guarantee that
  // the tasks will bind the same size data to them.
  if (shape_impl->unbound()) {
    throw detail::TracedException<std::invalid_argument>{
      "Shape of an unbound array or store cannot be used to create another array "
      "until the array or store is initialized by a task"};
  }
  return LogicalArray{
    impl_->create_array(std::move(shape_impl), type.impl(), nullable, optimize_scalar)};
}

LogicalArray Runtime::create_array_like(const LogicalArray& to_mirror, std::optional<Type> type)
{
  auto ty = type ? type.value().impl() : to_mirror.type().impl();

  return LogicalArray{impl_->create_array_like(to_mirror.impl(), std::move(ty))};
}

StringLogicalArray Runtime::create_string_array(const LogicalArray& descriptor,
                                                const LogicalArray& vardata)
{
  return LogicalArray{
    impl_->create_list_array(detail::string_type(), descriptor.impl(), vardata.impl())}
    .as_string_array();
}

ListLogicalArray Runtime::create_list_array(const LogicalArray& descriptor,
                                            const LogicalArray& vardata,
                                            std::optional<Type> ty /*=std::nullopt*/)
{
  auto type =
    ty ? ty->impl() : static_pointer_cast<detail::Type>(detail::list_type(vardata.type().impl()));
  return LogicalArray{impl_->create_list_array(std::move(type), descriptor.impl(), vardata.impl())}
    .as_list_array();
}

LogicalStore Runtime::create_store(const Type& type, std::uint32_t dim)
{
  return LogicalStore{impl_->create_store(
    make_internal_shared<detail::Shape>(dim), type.impl(), false /*optimize_scalar*/)};
}

LogicalStore Runtime::create_store(const Shape& shape,
                                   const Type& type,
                                   bool optimize_scalar /*=false*/)
{
  auto shape_impl = shape.impl();
  // We shouldn't allow users to create unbound store out of the same shape that hasn't be bound
  // yet. (See the comments in Runtime::create_array.)
  if (shape_impl->unbound()) {
    throw detail::TracedException<std::invalid_argument>{
      "Shape of an unbound array or store cannot be used to create another store "
      "until the array or store is initialized by a task"};
  }
  return LogicalStore{impl_->create_store(std::move(shape_impl), type.impl(), optimize_scalar)};
}

LogicalStore Runtime::create_store(const Scalar& scalar, const Shape& shape)
{
  return LogicalStore{impl_->create_store(*scalar.impl(), shape.impl())};
}

LogicalStore Runtime::create_store(const Shape& shape,
                                   const Type& type,
                                   void* buffer,
                                   bool read_only,
                                   const mapping::DimOrdering& ordering)
{
  const auto size = shape.volume() * type.size();
  return create_store(
    shape, type, ExternalAllocation::create_sysmem(buffer, size, read_only), ordering);
}

LogicalStore Runtime::create_store(const Shape& shape,
                                   const Type& type,
                                   const ExternalAllocation& allocation,
                                   const mapping::DimOrdering& ordering)
{
  return LogicalStore{
    impl_->create_store(shape.impl(), type.impl(), allocation.impl(), ordering.impl())};
}

std::pair<LogicalStore, LogicalStorePartition> Runtime::create_store(
  const Shape& shape,
  const tuple<std::uint64_t>& tile_shape,
  const Type& type,
  const std::vector<std::pair<ExternalAllocation, tuple<std::uint64_t>>>& allocations,
  const mapping::DimOrdering& ordering)
{
  auto [store, partition] =
    impl_->create_store(shape.impl(), tile_shape, type.impl(), allocations, ordering.impl());
  return {LogicalStore{std::move(store)}, LogicalStorePartition{std::move(partition)}};
}

void Runtime::prefetch_bloated_instances(const LogicalStore& store,
                                         tuple<std::uint64_t> low_offsets,
                                         tuple<std::uint64_t> high_offsets,
                                         bool initialize)
{
  impl_->prefetch_bloated_instances(
    store.impl(), std::move(low_offsets), std::move(high_offsets), initialize);
}

void Runtime::issue_mapping_fence() { impl_->issue_mapping_fence(); }

void Runtime::issue_execution_fence(bool block /*=false*/) { impl_->issue_execution_fence(block); }

void Runtime::raise_pending_exception() { impl_->raise_pending_exception(); }

std::uint32_t Runtime::node_count() const { return impl_->node_count(); }

std::uint32_t Runtime::node_id() const { return impl_->node_id(); }

void Runtime::register_shutdown_callback_(ShutdownCallback callback)
{
  detail::Runtime::get_runtime()->register_shutdown_callback(std::move(callback));
}

mapping::Machine Runtime::get_machine() const { return Scope::machine(); }

Processor Runtime::get_executing_processor() const { return impl()->get_executing_processor(); }

namespace {

std::optional<Runtime> the_public_runtime{};

}  // namespace

/*static*/ Runtime* Runtime::get_runtime()
{
  if (!has_started()) {
    throw detail::TracedException<std::runtime_error>{
      "Legate runtime has not been initialized. Please invoke legate::start to use the "
      "runtime"};
  }

  if (LEGATE_UNLIKELY(!the_public_runtime.has_value())) {
    the_public_runtime.emplace(Runtime{detail::Runtime::get_runtime()});
  }
  return &*the_public_runtime;
}

std::int32_t start(std::int32_t argc, char** argv)
{
  if (has_started()) {
    return 0;
  }

  if (has_finished()) {
    throw detail::TracedException<std::runtime_error>{
      "Legate runtime cannot be started after legate::finish is called"};
  }

  auto ret = detail::Runtime::start(argc, argv);
  // Called for effect, otherwise the following:
  //
  // legate::start(...);
  // legate::finish();
  //
  // Would never finalize the runtime, since the_public_runtime would never have been
  // constructed, so finish() would just return 0
  if (ret == 0) {
    static_cast<void>(Runtime::get_runtime());
  }
  return ret;
}

bool has_started() { return detail::has_started(); }

bool has_finished() { return detail::has_finished(); }

std::int32_t finish()
{
  if (!has_started()) {
    return 0;
  }

  const auto ret = Runtime::get_runtime()->impl()->finish();
  the_public_runtime.reset();
  return ret;
}

void destroy()
{
  if (const auto ret = finish()) {
    throw detail::TracedException<std::runtime_error>{
      fmt::format("failed to finalize legate runtime, error code: {}", ret)};
  }
}

mapping::Machine get_machine() { return Runtime::get_runtime()->get_machine(); }

bool is_running_in_task()
{
  // Make sure Legion runtime has been started and that we are not running in a user-thread
  // without a Legion context
  constexpr auto is_running_in_inline_task = [] {
    return has_started() && detail::Runtime::get_runtime()->executing_inline_task();
  };
  constexpr auto is_running_in_legion_task = [] {
    if (Legion::Runtime::has_runtime() && Legion::Runtime::has_context()) {
      return Legion::Runtime::get_context_task(Legion::Runtime::get_context())->has_parent_task();
    }
    return false;
  };

  return is_running_in_inline_task() || is_running_in_legion_task();
}

}  // namespace legate
