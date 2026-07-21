/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include <mutex>
#include <stdexcept>

namespace legate::detail {

namespace {

std::int64_t generate_library_reduction_ids(std::int64_t max_reduction_ops)
{
  // Reduction op IDs below this value are reserved for the core library; per-library
  // reduction ops are handed out starting from here.
  constexpr auto FIRST_LIBRARY_REDUCTION_ID = std::int64_t{100};
  static auto base                          = FIRST_LIBRARY_REDUCTION_ID;

  const auto result = base;

  base += max_reduction_ops;

  return result;
}

}  // namespace

Library::Library(std::string library_name,
                 const ResourceConfig& config,
                 std::unique_ptr<mapping::Mapper> mapper,
                 std::map<VariantCode, VariantOptions> default_options)
  : library_name_{std::move(library_name)},
    task_scope_{/*base=*/0, config.max_tasks, config.max_dyn_tasks},
    redop_scope_{generate_library_reduction_ids(config.max_reduction_ops),
                 config.max_reduction_ops},
    mapper_{std::move(mapper)},
    default_options_{std::move(default_options)}
{
  LEGATE_CHECK(mapper_ != nullptr);
}

// ==========================================================================================

GlobalRedopID Library::get_reduction_op_id(LocalRedopID local_redop_id) const
{
  return static_cast<GlobalRedopID>(
    redop_scope_.translate(static_cast<std::int64_t>(local_redop_id)));
}

LocalRedopID Library::get_local_reduction_op_id(GlobalRedopID redop_id) const
{
  return static_cast<LocalRedopID>(redop_scope_.invert(static_cast<std::int64_t>(redop_id)));
}

bool Library::valid_reduction_op_id(GlobalRedopID redop_id) const
{
  return redop_scope_.in_scope(static_cast<std::int64_t>(redop_id));
}

std::string_view Library::get_task_name(LocalTaskID local_task_id) const
{
  return find_task(local_task_id)->name().as_string_view();
}

void Library::register_task(
  LocalTaskID local_task_id,
  // try_emplace() doesn't unconditionally move, so clang-tidy thinks this is copied.
  InternalSharedPtr<TaskInfo> task_info  // NOLINT(performance-unnecessary-value-param)
)
{
  // Validate that the local task id falls within this library's task scope.
  if (!task_scope_.in_scope(static_cast<std::int64_t>(local_task_id))) {
    throw TracedException<std::out_of_range>{
      fmt::format("Task {} is invalid for library '{}' (max local task id: {})",
                  local_task_id,
                  get_library_name(),
                  task_scope_.size() - 1)};
  }

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "[" << get_library_name() << "] task " << to_underlying(local_task_id)
                         << ", " << legate::TaskInfo{task_info};
  }

  const auto [it, inserted] = tasks_.try_emplace(local_task_id, std::move(task_info));

  if (!inserted) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Task {} already exists in library {}", local_task_id, get_library_name())};
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
