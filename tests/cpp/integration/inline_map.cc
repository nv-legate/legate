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

#include "core/runtime/detail/runtime.h"
#include "legate.h"

namespace inline_map {

static const char* library_name = "test_inline_map";

enum TaskOpCode {
  ADDER,
};

struct AdderTask : public legate::LegateTask<AdderTask> {
  static const int32_t TASK_ID = ADDER;
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& output = context.outputs()[0];
    auto shape   = output.shape<1>();
    auto acc     = output.read_write_accessor<int64_t, 1>(shape);
    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) acc[*it] += 1;
  }
};

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  AdderTask::register_variants(context);
}

void test_mapped_regions_leak(legate::Runtime* runtime, legate::LibraryContext* context)
{
  {
    auto l_store = runtime->create_store({5}, legate::int64());
    auto p_store = l_store.get_physical_store(context);
    EXPECT_FALSE(p_store->is_future());
    EXPECT_EQ(runtime->impl()->num_inline_mapped(), 1);
  }
  EXPECT_EQ(runtime->impl()->num_inline_mapped(), 0);
}

void test_inline_map_future(legate::Runtime* runtime, legate::LibraryContext* context)
{
  auto l_store = runtime->create_store({1}, legate::int64(), true /*optimize_scalar*/);
  auto p_store = l_store.get_physical_store(context);
  EXPECT_TRUE(p_store->is_future());
}

void test_inline_map_region_and_slice(legate::Runtime* runtime, legate::LibraryContext* context)
{
  auto root_ls = runtime->create_store({5}, legate::int64());
  auto root_ps = root_ls.get_physical_store(context);
  EXPECT_FALSE(root_ps->is_future());
  auto slice_ls = root_ls.slice(0, legate::Slice(1));
  auto slice_ps = slice_ls.get_physical_store(context);
  EXPECT_FALSE(slice_ps->is_future());
  auto root_acc  = root_ps->write_accessor<int64_t, 1>();
  root_acc[2]    = 42;
  auto slice_acc = slice_ps->read_accessor<int64_t, 1>();
  EXPECT_EQ(slice_acc[1], 42);
}

void test_inline_map_and_task(legate::Runtime* runtime, legate::LibraryContext* context)
{
  auto l_store = runtime->create_store({5}, legate::int64());
  {
    auto p_store = l_store.get_physical_store(context);
    auto acc     = p_store->write_accessor<int64_t, 1>();
    acc[2]       = 42;
  }
  auto task = runtime->create_task(context, ADDER, {1});
  task->add_input(l_store);
  task->add_output(l_store);
  runtime->submit(std::move(task));
  auto p_store = l_store.get_physical_store(context);
  auto acc     = p_store->read_accessor<int64_t, 1>();
  EXPECT_EQ(acc[2], 43);
}

TEST(Integration, InlineMap)
{
  auto runtime = legate::Runtime::get_runtime();
  legate::Core::perform_registration<register_tasks>();
  auto context = runtime->find_library(library_name);

  test_mapped_regions_leak(runtime, context);
  test_inline_map_future(runtime, context);
  test_inline_map_region_and_slice(runtime, context);
  test_inline_map_and_task(runtime, context);
}

}  // namespace inline_map
