/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/runtime.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/data/detail/shape.h>
#include <legate/data/physical_store.h>
#include <legate/operation/detail/task.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/tuning/scope.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <fmt/core.h>

#include <cstdint>
#include <optional>
#include <stdexcept>

namespace legate {

Library Runtime::find_library(std::string_view library_name) const
{
  if (auto&& lib = impl_->find_library(std::move(library_name)); lib.has_value()) {
    return Library{&lib->get()};
  }
  throw detail::TracedException<std::out_of_range>{
    fmt::format("Library {} does not exist", library_name)};
}

std::optional<Library> Runtime::maybe_find_library(std::string_view library_name) const
{
  if (auto&& result = impl_->find_library(std::move(library_name)); result.has_value()) {
    return {Library{&result->get()}};
  }
  return std::nullopt;
}

Library Runtime::create_library(std::string_view library_name,
                                const ResourceConfig& config,
                                std::unique_ptr<mapping::Mapper> mapper,
                                std::map<VariantCode, VariantOptions> default_options)
{
  return Library{&impl_->create_library(
    std::move(library_name), config, std::move(mapper), std::move(default_options))};
}

Library Runtime::find_or_create_library(
  std::string_view library_name,
  const ResourceConfig& config,
  std::unique_ptr<mapping::Mapper> mapper,
  const std::map<VariantCode, VariantOptions>& default_options,
  bool* created)
{
  return Library{&impl_->find_or_create_library(
    std::move(library_name), config, std::move(mapper), default_options, created)};
}

AutoTask Runtime::create_task(Library library, LocalTaskID task_id)
{
  return AutoTask{impl_->create_task(*library.impl(), task_id)};
}

ManualTask Runtime::create_task(Library library,
                                LocalTaskID task_id,
                                const tuple<std::uint64_t>& launch_shape)
{
  return create_task(library, task_id, launch_shape.data());
}

ManualTask Runtime::create_task(Library library,
                                LocalTaskID task_id,
                                Span<const std::uint64_t> launch_shape)
{
  return create_task(library, task_id, detail::to_domain(launch_shape));
}

ManualTask Runtime::create_task(Library library,
                                LocalTaskID task_id,
                                std::initializer_list<std::uint64_t> launch_shape)
{
  return create_task(library, task_id, Span<const std::uint64_t>{launch_shape});
}

ManualTask Runtime::create_task(Library library, LocalTaskID task_id, const Domain& launch_domain)
{
  return ManualTask{impl_->create_task(*library.impl(), task_id, launch_domain)};
}

void Runtime::issue_fill(const LogicalStore& lhs, const LogicalStore& value)
{
  impl_->issue_fill(lhs.impl(), value.impl());
}

void Runtime::issue_fill(const LogicalStore& lhs, const Scalar& value)
{
  impl_->issue_fill(lhs.impl(), *value.impl());
}

void Runtime::submit(AutoTask&& task) { impl_->submit(task.impl_()); }

void Runtime::submit(ManualTask&& task) { impl_->submit(task.impl_()); }

LogicalStore Runtime::create_store(const Type& type, std::uint32_t dim)
{
  return LogicalStore{impl_->create_store(make_internal_shared<detail::Shape>(dim), type.impl())};
}

LogicalStore Runtime::create_store(const Shape& shape, const Type& type)
{
  auto shape_impl = shape.impl();
  // We shouldn't allow users to create unbound store out of the same shape that hasn't be bound
  // yet. (See the comments in Runtime::create_array.)
  if (shape_impl->unbound()) {
    throw detail::TracedException<std::invalid_argument>{
      "Shape of an unbound array or store cannot be used to create another store "
      "until the array or store is initialized by a task"};
  }
  return LogicalStore{impl_->create_store(std::move(shape_impl), type.impl())};
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
    impl_->create_store(shape.impl(),
                        {detail::tags::iterator_tag, tile_shape.begin(), tile_shape.end()},
                        type.impl(),
                        allocations,
                        ordering.impl());
  return {LogicalStore{std::move(store)}, LogicalStorePartition{std::move(partition)}};
}

void Runtime::issue_mapping_fence() { impl_->issue_mapping_fence(); }

void Runtime::issue_execution_fence(bool block /*=false*/) { impl_->issue_execution_fence(block); }

std::uint32_t Runtime::node_count() const { return impl_->node_count(); }

std::uint32_t Runtime::node_id() const { return impl_->node_id(); }

void Runtime::register_shutdown_callback_(ShutdownCallback callback)
{
  detail::Runtime::get_runtime().register_shutdown_callback(std::move(callback));
}

mapping::Machine Runtime::get_machine() const { return Scope::machine(); }

Processor Runtime::get_executing_processor() const { return impl()->get_executing_processor(); }

void* Runtime::get_cuda_stream() const { return impl()->get_cuda_stream(); }

CUdevice Runtime::get_current_cuda_device() const { return impl()->get_current_cuda_device(); }

// This method can be static in theory, but is made a proper member so users can call this only
// after the runtime is started.
void Runtime::synchronize_cuda_stream(
  void* stream) const  // NOLINT(readability-convert-member-functions-to-static)
{
  cuda::detail::get_cuda_driver_api()->stream_synchronize(static_cast<CUstream>(stream));
}

const detail::Config& Runtime::config() const { return impl()->config(); }

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

void start()
{
  detail::Runtime::start();
  // Called for effect, otherwise the following:
  //
  // legate::start(...);
  // legate::finish();
  //
  // Would never finalize the runtime, since the_public_runtime would never have been
  // constructed, so finish() would just return 0
  static_cast<void>(Runtime::get_runtime());
}

std::int32_t start(std::int32_t, char**)
{
  start();
  return 0;
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

bool is_running_in_task() { return false; }

}  // namespace legate
