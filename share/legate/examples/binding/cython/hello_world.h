/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

namespace hello_world {

class HelloWorld : public legate::LegateTask<HelloWorld> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{5};

  static void cpu_variant(legate::TaskContext);
};

}  // namespace hello_world
