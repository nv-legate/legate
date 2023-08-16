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

#pragma once

#include "core/mapping/mapping.h"
#include "legate.h"

namespace task {

namespace hello {

enum TaskOpCode {
  _OP_CODE_BASE = 0,
  HELLO_WORLD   = 1,
  SUM           = 2,
  SQUARE        = 3,
  IOTA          = 4,
};

static const char* library_name = "legate.hello";
extern Legion::Logger logger;

void register_tasks();

struct HelloWorldTask : public legate::LegateTask<HelloWorldTask> {
  static const int32_t TASK_ID = HELLO_WORLD;
  static void cpu_variant(legate::TaskContext context);
};

struct SumTask : public legate::LegateTask<SumTask> {
  static const int32_t TASK_ID = SUM;
  static void cpu_variant(legate::TaskContext context);
};

struct SquareTask : public legate::LegateTask<SquareTask> {
  static const int32_t TASK_ID = SQUARE;
  static void cpu_variant(legate::TaskContext context);
};

struct IotaTask : public legate::LegateTask<IotaTask> {
  static const int32_t TASK_ID = IOTA;
  static void cpu_variant(legate::TaskContext context);
};

}  // namespace hello

}  // namespace task
