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

#include "core/task/registrar.h"

#include "core/runtime/detail/library.h"
#include "core/task/task_info.h"

#include <utility>
#include <vector>

namespace legate {

struct TaskRegistrar::Impl {
  std::vector<std::pair<int64_t, std::unique_ptr<TaskInfo>>> pending_task_infos{};
};

void TaskRegistrar::record_task(int64_t local_task_id, std::unique_ptr<TaskInfo> task_info)
{
  impl_->pending_task_infos.emplace_back(local_task_id, std::move(task_info));
}

void TaskRegistrar::register_all_tasks(Library library)
{
  auto* lib_impl = library.impl();
  for (auto& [local_task_id, task_info] : impl_->pending_task_infos) {
    lib_impl->register_task(local_task_id, std::move(task_info));
  }
  impl_->pending_task_infos.clear();
}

TaskRegistrar::TaskRegistrar() : impl_{std::make_unique<TaskRegistrar::Impl>()} {}

TaskRegistrar::~TaskRegistrar() = default;

}  // namespace legate
