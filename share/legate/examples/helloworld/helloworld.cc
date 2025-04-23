/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <iostream>
#include <string_view>

inline constexpr std::string_view LIBRARY_NAME = "helloworld";

class HelloTask : public legate::LegateTask<HelloTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext)
  {
    std::cout << "HelloTask::cpu_variant ran" << std::endl;
  }
};

int main()
{
  std::cout << "Helloworld started" << std::endl;

  legate::start();

  auto* runtime = legate::Runtime::get_runtime();
  auto library  = runtime->create_library(LIBRARY_NAME);
  HelloTask::register_variants(library);

  auto task = runtime->create_task(library, HelloTask::TASK_CONFIG.task_id());
  runtime->submit(std::move(task));

  auto status = legate::finish();
  std::cout << "Helloworld finished" << std::endl;

  return status;
}
