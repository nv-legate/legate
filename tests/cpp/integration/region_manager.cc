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

#include <gtest/gtest.h>

#include "legate.h"

namespace region_manager {

namespace {

static const char* library_name = "test_region_manager";

struct TesterTask : public legate::LegateTask<TesterTask> {
  static const int32_t TASK_ID = 0;
  static void cpu_variant(legate::TaskContext& context) {}
};

void prepare()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  TesterTask::register_variants(context);
}

}  // namespace

TEST(Integration, RegionManager)
{
  legate::Core::perform_registration<prepare>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, 0);

  std::vector<legate::LogicalStore> stores;
  for (uint32_t idx = 0; idx < LEGION_MAX_FIELDS * 2; ++idx) {
    auto store = runtime->create_store({10}, legate::int64());
    auto part  = task->declare_partition();
    task->add_output(store, part);
    stores.push_back(store);
  }
  runtime->submit(std::move(task));
}

}  // namespace region_manager
