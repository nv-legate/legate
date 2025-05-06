/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <iostream>
#include <string_view>

class HelloTask : public legate::LegateTask<HelloTask> {
 public:
  static constexpr std::string_view LIBRARY_NAME = "helloworld";
  static inline const auto TASK_CONFIG           =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext) { std::cout << "Hello, world!\n"; }
};

int main()
{
  legate::start();

  auto* const runtime = legate::Runtime::get_runtime();

  auto library = runtime->create_library(HelloTask::LIBRARY_NAME);

  HelloTask::register_variants(library);

  {
    auto task = runtime->create_task(library, HelloTask::TASK_CONFIG.task_id());

    runtime->submit(std::move(task));
  }

  return legate::finish();
}
