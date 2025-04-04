/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/registrar.h>

#include <legate/runtime/library.h>
#include <legate/task/task.h>
#include <legate/utilities/typedefs.h>

#include <functional>
#include <utility>
#include <vector>

namespace legate {

class TaskRegistrar::Impl {
 public:
  std::vector<std::function<void(const Library&)>> pending_{};
};

void TaskRegistrar::record_registration_function(RecordTaskKey,
                                                 std::function<void(const Library&)> func)
{
  impl_->pending_.emplace_back(std::move(func));
}

void TaskRegistrar::register_all_tasks(Library& library)
{
  for (auto&& func : impl_->pending_) {
    func(library);
  }
  impl_->pending_.clear();
}

TaskRegistrar::TaskRegistrar() : impl_{std::make_unique<TaskRegistrar::Impl>()} {}

TaskRegistrar::~TaskRegistrar() = default;

}  // namespace legate
