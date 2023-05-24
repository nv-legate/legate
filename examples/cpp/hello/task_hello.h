/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
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
  static void cpu_variant(legate::TaskContext& context);
};

struct SumTask : public legate::LegateTask<SumTask> {
  static const int32_t TASK_ID = SUM;
  static void cpu_variant(legate::TaskContext& context);
};

struct SquareTask : public legate::LegateTask<SquareTask> {
  static const int32_t TASK_ID = SQUARE;
  static void cpu_variant(legate::TaskContext& context);
};

struct IotaTask : public legate::LegateTask<IotaTask> {
  static const int32_t TASK_ID = IOTA;
  static void cpu_variant(legate::TaskContext& context);
};

}  // namespace hello

}  // namespace task
