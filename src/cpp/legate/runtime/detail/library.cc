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

#include <legate/runtime/detail/library.h>

#include <legate/mapping/mapping.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/runtime/runtime.h>
#include <legate/task/detail/task_info.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/type_traits.h>

#include <fmt/format.h>

#include <exception>
#include <stdexcept>

namespace legate::detail {

Library::Library(std::string library_name,
                 const ResourceConfig& config,
                 std::unique_ptr<mapping::Mapper> mapper,
                 std::map<VariantCode, VariantOptions> default_options,
                 Legion::Runtime* runtime)
  : library_name_{std::move(library_name)},
    task_scope_{runtime->generate_library_task_ids(library_name_.c_str(), config.max_tasks),
                config.max_tasks,
                config.max_dyn_tasks},
    redop_scope_{
      runtime->generate_library_reduction_ids(library_name_.c_str(), config.max_reduction_ops),
      config.max_reduction_ops},
    proj_scope_{
      runtime->generate_library_projection_ids(library_name_.c_str(), config.max_projections),
      config.max_projections},
    shard_scope_{
      runtime->generate_library_sharding_ids(library_name_.c_str(), config.max_shardings),
      config.max_shardings},
    mapper_{std::move(mapper)},
    default_options_{std::move(default_options)}
{
}

// ==========================================================================================

Library::Library(ConstructKey,
                 std::string library_name,
                 const ResourceConfig& config,
                 std::unique_ptr<mapping::Mapper> mapper,
                 std::map<VariantCode, VariantOptions> default_options)
  : Library{std::move(library_name),
            config,
            std::move(mapper),
            std::move(default_options),
            Legion::Runtime::get_runtime()}
{
}

GlobalTaskID Library::get_task_id(LocalTaskID local_task_id) const
{
  return static_cast<GlobalTaskID>(task_scope_.translate(static_cast<std::int64_t>(local_task_id)));
}

GlobalRedopID Library::get_reduction_op_id(LocalRedopID local_redop_id) const
{
  return static_cast<GlobalRedopID>(
    redop_scope_.translate(static_cast<std::int64_t>(local_redop_id)));
}

Legion::ProjectionID Library::get_projection_id(std::int64_t local_proj_id) const
{
  if (local_proj_id == 0) {
    return 0;
  }
  return static_cast<Legion::ProjectionID>(proj_scope_.translate(local_proj_id));
}

Legion::ShardingID Library::get_sharding_id(std::int64_t local_shard_id) const
{
  return static_cast<Legion::ShardingID>(shard_scope_.translate(local_shard_id));
}

LocalTaskID Library::get_local_task_id(GlobalTaskID task_id) const
{
  return static_cast<LocalTaskID>(task_scope_.invert(static_cast<std::int64_t>(task_id)));
}

LocalRedopID Library::get_local_reduction_op_id(GlobalRedopID redop_id) const
{
  return static_cast<LocalRedopID>(redop_scope_.invert(static_cast<std::int64_t>(redop_id)));
}

std::int64_t Library::get_local_projection_id(Legion::ProjectionID proj_id) const
{
  if (proj_id == 0) {
    return 0;
  }
  return proj_scope_.invert(proj_id);
}

std::int64_t Library::get_local_sharding_id(Legion::ShardingID shard_id) const
{
  return shard_scope_.invert(shard_id);
}

bool Library::valid_task_id(GlobalTaskID task_id) const
{
  return task_scope_.in_scope(static_cast<std::int64_t>(task_id));
}

bool Library::valid_reduction_op_id(GlobalRedopID redop_id) const
{
  return redop_scope_.in_scope(static_cast<std::int64_t>(redop_id));
}

bool Library::valid_projection_id(Legion::ProjectionID proj_id) const
{
  return proj_scope_.in_scope(proj_id);
}

bool Library::valid_sharding_id(Legion::ShardingID shard_id) const
{
  return shard_scope_.in_scope(shard_id);
}

std::string_view Library::get_task_name(LocalTaskID local_task_id) const
{
  return find_task(local_task_id)->name().as_string_view();
}

std::unique_ptr<Scalar> Library::get_tunable(std::int64_t tunable_id,
                                             InternalSharedPtr<Type> type) const
{
  if (type->variable_size()) {
    throw TracedException<std::invalid_argument>{"Tunable variables must have fixed-size types"};
  }
  auto result         = Runtime::get_runtime()->get_tunable(*this, tunable_id);
  std::size_t extents = 0;
  const void* buffer  = result.get_buffer(Memory::Kind::SYSTEM_MEM, &extents);
  if (extents != type->size()) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Size mismatch: expected {} bytes but got {} bytes", type->size(), extents)};
  }
  return std::make_unique<Scalar>(std::move(type), buffer, true);
}

void Library::register_task(
  LocalTaskID local_task_id,
  // try_emplace() doesn't unconditionally move, so clang-tidy thinks this is copied.
  InternalSharedPtr<TaskInfo> task_info  // NOLINT(performance-unnecessary-value-param)
)
{
  const auto task_id = [&] {
    try {
      return get_task_id(local_task_id);
    } catch (const std::out_of_range&) {
      throw TracedException<std::out_of_range>{
        fmt::format("Task {} is invalid for library '{}' (max local task id: {})",
                    local_task_id,
                    get_library_name(),
                    task_scope_.size() - 1)};
    }
  }();

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "[" << get_library_name() << "] task "
                         << traits::detail::to_underlying(local_task_id)
                         << " (global id: " << traits::detail::to_underlying(task_id) << "), "
                         << legate::TaskInfo{task_info};
  }

  const auto [it, inserted] = tasks_.try_emplace(local_task_id, std::move(task_info));

  if (!inserted) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Task {} already exists in library {}", local_task_id, get_library_name())};
  }

  try {
    it->second->register_task(task_id);
  } catch (...) {
    tasks_.erase(it);
    throw;
  }
}

const InternalSharedPtr<TaskInfo>& Library::find_task(LocalTaskID local_task_id) const
{
  const auto it = tasks_.find(local_task_id);

  if (tasks_.end() == it) {
    throw TracedException<std::out_of_range>{
      fmt::format("Library {} does not have task {}", get_library_name(), local_task_id)};
  }
  return it->second;
}

}  // namespace legate::detail
