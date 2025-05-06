/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <cstdint>
#include <iostream>

namespace hello_world {

class HelloWorld : public legate::LegateTask<HelloWorld> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext);
};

void HelloWorld::cpu_variant(legate::TaskContext) { std::cout << "Hello World!\n"; }

}  // namespace hello_world

extern "C" {

std::int64_t hello_world_task_id()
{
  return static_cast<std::int64_t>(hello_world::HelloWorld::TASK_CONFIG.task_id());
}

void hello_world_register_task(void* lib_ptr)
{
  hello_world::HelloWorld::register_variants(*static_cast<legate::Library*>(lib_ptr));
}

}  // extern "C"
