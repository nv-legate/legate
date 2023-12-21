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

#include "core/runtime/runtime.h"

#include "core/operation/detail/task.h"
#include "core/runtime/detail/runtime.h"

namespace legate {

Library Runtime::find_library(const std::string& library_name) const
{
  return Library{impl_->find_library(library_name, false)};
}

std::optional<Library> Runtime::maybe_find_library(const std::string& library_name) const
{
  if (auto result = impl_->find_library(library_name, true)) {
    return {Library{std::move(result)}};
  }
  return std::nullopt;
}

Library Runtime::create_library(const std::string& library_name,
                                const ResourceConfig& config,
                                std::unique_ptr<mapping::Mapper> mapper)
{
  return Library{
    impl_->create_library(library_name, config, std::move(mapper), false /*in_callback*/)};
}

Library Runtime::find_or_create_library(const std::string& library_name,
                                        const ResourceConfig& config,
                                        std::unique_ptr<mapping::Mapper> mapper,
                                        bool* created)
{
  return Library{impl_->find_or_create_library(
    library_name, config, std::move(mapper), created, false /*in_callback*/)};
}

AutoTask Runtime::create_task(Library library, int64_t task_id)
{
  return AutoTask{impl_->create_task(library.impl(), task_id)};
}

ManualTask Runtime::create_task(Library library, int64_t task_id, const Shape& launch_shape)
{
  return create_task(library, task_id, to_domain(launch_shape));
}

ManualTask Runtime::create_task(Library library, int64_t task_id, const Domain& launch_domain)
{
  return ManualTask{impl_->create_task(library.impl(), task_id, launch_domain)};
}

void Runtime::issue_copy(LogicalStore& target,
                         const LogicalStore& source,
                         std::optional<ReductionOpKind> redop)
{
  auto op = redop ? std::make_optional(static_cast<int32_t>(redop.value())) : std::nullopt;
  impl_->issue_copy(target.impl(), source.impl(), op);
}

void Runtime::issue_copy(LogicalStore& target,
                         const LogicalStore& source,
                         std::optional<int32_t> redop)
{
  impl_->issue_copy(target.impl(), source.impl(), redop);
}

void Runtime::issue_gather(LogicalStore& target,
                           const LogicalStore& source,
                           const LogicalStore& source_indirect,
                           std::optional<ReductionOpKind> redop)
{
  auto op = redop ? std::make_optional(static_cast<int32_t>(redop.value())) : std::nullopt;
  impl_->issue_gather(target.impl(), source.impl(), source_indirect.impl(), op);
}

void Runtime::issue_gather(LogicalStore& target,
                           const LogicalStore& source,
                           const LogicalStore& source_indirect,
                           std::optional<int32_t> redop)
{
  impl_->issue_gather(target.impl(), source.impl(), source_indirect.impl(), redop);
}

void Runtime::issue_scatter(LogicalStore& target,
                            const LogicalStore& target_indirect,
                            const LogicalStore& source,
                            std::optional<ReductionOpKind> redop)
{
  auto op = redop ? std::make_optional(static_cast<int32_t>(redop.value())) : std::nullopt;
  impl_->issue_scatter(target.impl(), target_indirect.impl(), source.impl(), op);
}

void Runtime::issue_scatter(LogicalStore& target,
                            const LogicalStore& target_indirect,
                            const LogicalStore& source,
                            std::optional<int32_t> redop)
{
  impl_->issue_scatter(target.impl(), target_indirect.impl(), source.impl(), redop);
}

void Runtime::issue_scatter_gather(LogicalStore& target,
                                   const LogicalStore& target_indirect,
                                   const LogicalStore& source,
                                   const LogicalStore& source_indirect,
                                   std::optional<ReductionOpKind> redop)
{
  auto op = redop ? std::make_optional(static_cast<int32_t>(redop.value())) : std::nullopt;
  impl_->issue_scatter_gather(
    target.impl(), target_indirect.impl(), source.impl(), source_indirect.impl(), op);
}

void Runtime::issue_scatter_gather(LogicalStore& target,
                                   const LogicalStore& target_indirect,
                                   const LogicalStore& source,
                                   const LogicalStore& source_indirect,
                                   std::optional<int32_t> redop)
{
  impl_->issue_scatter_gather(
    target.impl(), target_indirect.impl(), source.impl(), source_indirect.impl(), redop);
}

void Runtime::issue_fill(const LogicalArray& lhs, const LogicalStore& value)
{
  impl_->issue_fill(lhs.impl(), value.impl());
}

void Runtime::issue_fill(const LogicalArray& lhs, const Scalar& value)
{
  issue_fill(lhs, create_store(value));
}

LogicalStore Runtime::tree_reduce(Library library,
                                  int64_t task_id,
                                  const LogicalStore& store,
                                  int32_t radix)
{
  auto out_store = create_store(store.type(), 1);

  impl_->tree_reduce(library.impl(), task_id, store.impl(), out_store.impl(), radix);
  return out_store;
}

void Runtime::submit(AutoTask task) { impl_->submit(std::move(task.impl_)); }

void Runtime::submit(ManualTask task) { impl_->submit(std::move(task.impl_)); }

LogicalArray Runtime::create_array(const Type& type, uint32_t dim, bool nullable)
{
  return LogicalArray{impl_->create_array(type.impl(), dim, nullable)};
}

LogicalArray Runtime::create_array(const Shape& extents,
                                   const Type& type,
                                   bool nullable,
                                   bool optimize_scalar)
{
  return LogicalArray{impl_->create_array(extents, type.impl(), nullable, optimize_scalar)};
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
  auto type = ty ? ty->impl() : detail::list_type(vardata.type().impl());
  return LogicalArray{impl_->create_list_array(std::move(type), descriptor.impl(), vardata.impl())}
    .as_list_array();
}

LogicalStore Runtime::create_store(const Type& type, uint32_t dim)
{
  return LogicalStore{impl_->create_store(type.impl(), dim)};
}

LogicalStore Runtime::create_store(const Shape& extents,
                                   const Type& type,
                                   bool optimize_scalar /*=false*/)
{
  return LogicalStore{impl_->create_store(extents, type.impl(), optimize_scalar)};
}

LogicalStore Runtime::create_store(const Scalar& scalar, const Shape& extents)
{
  return LogicalStore{impl_->create_store(*scalar.impl(), extents)};
}

LogicalStore Runtime::create_store(const Shape& extents,
                                   const Type& type,
                                   void* buffer,
                                   bool read_only,
                                   const mapping::DimOrdering& ordering)
{
  const auto size = extents.volume() * type.size();
  return create_store(
    extents, type, ExternalAllocation::create_sysmem(buffer, size, read_only), ordering);
}

LogicalStore Runtime::create_store(const Shape& extents,
                                   const Type& type,
                                   const ExternalAllocation& allocation,
                                   const mapping::DimOrdering& ordering)
{
  return LogicalStore{
    impl_->create_store(extents, type.impl(), allocation.impl(), ordering.impl())};
}

std::pair<LogicalStore, LogicalStorePartition> Runtime::create_store(
  const Shape& extents,
  const Shape& tile_shape,
  const Type& type,
  const std::vector<std::pair<ExternalAllocation, Shape>>& allocations,
  const mapping::DimOrdering& ordering)
{
  auto [store, partition] =
    impl_->create_store(extents, tile_shape, type.impl(), allocations, ordering.impl());
  return {LogicalStore{std::move(store)}, LogicalStorePartition{std::move(partition)}};
}

void Runtime::issue_execution_fence(bool block /*=false*/) { impl_->issue_execution_fence(block); }

void Runtime::register_shutdown_callback_(ShutdownCallback callback)
{
  detail::Runtime::get_runtime()->register_shutdown_callback(std::move(callback));
}

mapping::Machine Runtime::get_machine() const { return mapping::Machine{impl_->get_machine()}; }

/*static*/ Runtime* Runtime::get_runtime()
{
  static Runtime* the_runtime{};

  if (!the_runtime) {
    auto* impl = detail::Runtime::get_runtime();

    if (!impl->initialized()) {
      throw std::runtime_error{
        "Legate runtime has not been initialized. Please invoke legate::start to use the runtime"};
    }

    the_runtime = new Runtime{impl};
  }
  return the_runtime;
}

int32_t start(int32_t argc, char** argv) { return detail::Runtime::start(argc, argv); }

int32_t finish() { return Runtime::get_runtime()->impl()->finish(); }

void destroy() { detail::Runtime::get_runtime()->destroy(); }

mapping::Machine get_machine() { return Runtime::get_runtime()->get_machine(); }

}  // namespace legate

extern "C" {

void legate_core_perform_registration()
{
  // Tell the runtime about our registration callback so we can register ourselves
  // Make sure it is global so this shared object always gets loaded on all nodes
  Legion::Runtime::perform_registration_callback(legate::detail::initialize_core_library_callback,
                                                 true /*global*/);
  legate::detail::Runtime::get_runtime()->initialize(Legion::Runtime::get_context());
}
}
