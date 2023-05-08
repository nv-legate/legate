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

#include "core/mapping/mapping.h"
#include "legate.h"

namespace manualsimple {

static const char* library_name = "manual";
static legate::Logger logger(library_name);

enum TaskIDs {
  HELLO = 0,
};

struct HelloTask : public legate::LegateTask<HelloTask> {
  static const int32_t TASK_ID = HELLO;
  static void cpu_variant(legate::TaskContext& context);
};

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  HelloTask::register_variants(context);
}

/*static*/ void HelloTask::cpu_variant(legate::TaskContext& context)
{
  auto& output = context.outputs()[0];
  auto shape   = output.shape<2>();

  if (shape.empty()) return;

  auto acc = output.write_accessor<int64_t, 2>(shape);
  for (legate::PointInRectIterator<2> it(shape); it.valid(); ++it)
    acc[*it] = (*it)[0] + (*it)[1] * 1000;
}

void test_auto_task(legate::LibraryContext* context, legate::LogicalStore store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(context, HELLO);
  auto part    = task->declare_partition();
  task->add_output(store, part);
  runtime->submit(std::move(task));
}

void test_manual_task(legate::LibraryContext* context, legate::LogicalStore store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(context, HELLO, {3, 3});
  auto part    = store.partition_by_tiling({2, 2});
  task->add_output(part);
  runtime->submit(std::move(task));
}

void print_store(legate::LibraryContext* context, legate::LogicalStore store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto p_store = store.get_physical_store(context);
  auto acc     = p_store->read_accessor<int64_t, 2>();
  auto shape   = p_store->shape<2>();
  std::stringstream ss;
  for (legate::PointInRectIterator<2> it(shape); it.valid(); ++it) ss << acc[*it] << " ";
  logger.print() << ss.str();
}

void legate_main(int32_t argc, char** argv)
{
  legate::Core::perform_registration<register_tasks>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto store = runtime->create_store({5, 5}, legate::int64());
  test_auto_task(context, store);
  print_store(context, store);
  test_manual_task(context, store);
  print_store(context, store);
}

}  // namespace manualsimple

TEST(Integration, ManualSimple)
{
  legate::initialize(0, NULL);
  legate::set_main_function(manualsimple::legate_main);
  legate::start(0, NULL);
}
