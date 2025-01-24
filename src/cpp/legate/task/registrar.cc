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

#include <legate/task/registrar.h>

#include <legate/runtime/detail/library.h>
#include <legate/task/task_info.h>

#include <utility>
#include <vector>

namespace legate {

class TaskRegistrar::Impl {
 public:
  std::vector<std::pair<LocalTaskID, std::function<std::unique_ptr<TaskInfo>(const Library&)>>>
    pending_task_infos{};
};

void TaskRegistrar::record_task(LocalTaskID local_task_id, std::unique_ptr<TaskInfo> task_info)
{
  record_task(RecordTaskKey{},
              local_task_id,
              // This workaround is needed because std::function requires the callable to be
              // copy-assignable. Any object which holds a unique_ptr, is by definition not
              // copy-assignable, so we work around this by putting our unique_ptr in a
              // shared_ptr.  But make no mistake, this callable is only callable once
              [tinfo = std::make_shared<std::unique_ptr<TaskInfo>>(std::move(task_info))](
                const Library&) { return std::move(*tinfo); });
}

void TaskRegistrar::record_task(
  RecordTaskKey,
  LocalTaskID local_task_id,
  std::function<std::unique_ptr<TaskInfo>(const Library&)> deferred_task_info)
{
  impl_->pending_task_infos.emplace_back(local_task_id, std::move(deferred_task_info));
}

void TaskRegistrar::register_all_tasks(Library& library)
{
  auto* lib_impl = library.impl();

  for (auto&& [local_task_id, task_info_fn] : impl_->pending_task_infos) {
    lib_impl->register_task(local_task_id, task_info_fn(library));
  }
  impl_->pending_task_infos.clear();
}

TaskRegistrar::TaskRegistrar() : impl_{std::make_unique<TaskRegistrar::Impl>()} {}

TaskRegistrar::~TaskRegistrar() = default;

}  // namespace legate
