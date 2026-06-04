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
    mapper_{std::move(mapper)},
    default_options_{std::move(default_options)}
{
  LEGATE_CHECK(mapper_ != nullptr);
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

/*static*/ void Library::perform_callback(Legion::RegistrationWithArgsCallbackFnptr callback,
                                          const Legion::UntypedBuffer& buffer)
{
  // Under single-controller execution, only the controller will call this method; however,
  // callback must be executed on all workers, so register the callback globally.
  const bool global = Runtime::get_runtime().config().single_controller_execution();

  Legion::Runtime::perform_registration_callback(
    callback, buffer, global /*global*/, false /*dedup*/);
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

LocalTaskID Library::get_local_task_id(GlobalTaskID task_id) const
{
  return static_cast<LocalTaskID>(task_scope_.invert(static_cast<std::int64_t>(task_id)));
}

LocalRedopID Library::get_local_reduction_op_id(GlobalRedopID redop_id) const
{
  return static_cast<LocalRedopID>(redop_scope_.invert(static_cast<std::int64_t>(redop_id)));
}

bool Library::valid_task_id(GlobalTaskID task_id) const
{
  return task_scope_.in_scope(static_cast<std::int64_t>(task_id));
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
    log_legate().debug() << "[" << get_library_name() << "] task " << to_underlying(local_task_id)
                         << " (global id: " << to_underlying(task_id) << "), "
                         << legate::TaskInfo{task_info};
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
