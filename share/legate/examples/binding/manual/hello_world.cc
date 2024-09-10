/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cstdint>
#include <iostream>
#include <legate.h>

namespace hello_world {

class HelloWorld : public legate::LegateTask<HelloWorld> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  static void cpu_variant(legate::TaskContext);
};

void HelloWorld::cpu_variant(legate::TaskContext) { std::cout << "Hello World!\n"; }

}  // namespace hello_world

extern "C" {

std::int64_t hello_world_task_id()
{
  return static_cast<std::int64_t>(hello_world::HelloWorld::TASK_ID);
}

void hello_world_register_task(void* lib_ptr)
{
  hello_world::HelloWorld::register_variants(*static_cast<legate::Library*>(lib_ptr));
}

}  // extern "C"
