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

#include <legate/runtime/library.h>
#include <legate/task/task_info.h>
#include <legate/utilities/typedefs.h>

#include <utility>
#include <vector>

namespace legate {

class TaskRegistrar::Impl {
 public:
  std::vector<std::pair<LocalTaskID, std::function<TaskInfo(const Library&)>>> pending_task_infos{};
};

void TaskRegistrar::record_task(RecordTaskKey,
                                LocalTaskID local_task_id,
                                std::function<TaskInfo(const Library&)> deferred_task_info)
{
  impl_->pending_task_infos.emplace_back(local_task_id, std::move(deferred_task_info));
}

void TaskRegistrar::register_all_tasks(Library& library)
{
  for (auto&& [local_task_id, task_info_fn] : impl_->pending_task_infos) {
    library.register_task(local_task_id, task_info_fn(library));
  }
  impl_->pending_task_infos.clear();
}

TaskRegistrar::TaskRegistrar() : impl_{std::make_unique<TaskRegistrar::Impl>()} {}

TaskRegistrar::~TaskRegistrar() = default;

}  // namespace legate
