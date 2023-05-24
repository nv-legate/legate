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

namespace simple {

enum TaskOpCode {
  _OP_CODE_BASE = 0,
  HELLO         = 1,
  WRITER        = 2,
  REDUCER       = 3,
};

static const char* library_name = "legate.simple";
extern Legion::Logger logger;

void register_tasks();

struct HelloTask : public legate::LegateTask<HelloTask> {
  static const int32_t TASK_ID = HELLO;
  static void cpu_variant(legate::TaskContext& context);
};

struct WriterTask : public legate::LegateTask<WriterTask> {
  static const int32_t TASK_ID = WRITER;
  static void cpu_variant(legate::TaskContext& context);
};

struct ReducerTask : public legate::LegateTask<ReducerTask> {
  static const int32_t TASK_ID = REDUCER;
  static void cpu_variant(legate::TaskContext& context);
};

}  // namespace simple

}  // namespace task
