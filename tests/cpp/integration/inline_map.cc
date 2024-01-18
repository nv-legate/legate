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

#include "core/runtime/detail/runtime.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace inline_map {

using InlineMap = DefaultFixture;

static const char* library_name = "test_inline_map";

enum TaskOpCode {
  ADDER,
};

struct AdderTask : public legate::LegateTask<AdderTask> {
  static const int32_t TASK_ID = ADDER;
  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0).data();
    auto shape  = output.shape<1>();
    auto acc    = output.read_write_accessor<int64_t, 1>(shape);
    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
      acc[*it] += 1;
    }
  }
};

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  AdderTask::register_variants(context);
}

void test_inline_map_future()
{
  auto runtime = legate::Runtime::get_runtime();
  auto l_store = runtime->create_store(legate::Shape{1}, legate::int64(), true /*optimize_scalar*/);
  auto p_store = l_store.get_physical_store();
  EXPECT_TRUE(p_store.is_future());
}

void test_inline_map_region_and_slice()
{
  auto runtime = legate::Runtime::get_runtime();
  auto root_ls = runtime->create_store(legate::Shape{5}, legate::int64());
  auto root_ps = root_ls.get_physical_store();
  EXPECT_FALSE(root_ps.is_future());
  auto slice_ls = root_ls.slice(0, legate::Slice(1));
  auto slice_ps = slice_ls.get_physical_store();
  EXPECT_FALSE(slice_ps.is_future());
  auto root_acc  = root_ps.write_accessor<int64_t, 1>();
  root_acc[2]    = 42;
  auto slice_acc = slice_ps.read_accessor<int64_t, 1>();
  EXPECT_EQ(slice_acc[1], 42);
}

void test_inline_map_and_task()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto l_store = runtime->create_store(legate::Shape{5}, legate::int64());
  {
    auto p_store = l_store.get_physical_store();
    auto acc     = p_store.write_accessor<int64_t, 1>();
    acc[2]       = 42;
  }
  auto task = runtime->create_task(context, ADDER, {1});
  task.add_input(l_store);
  task.add_output(l_store);
  runtime->submit(std::move(task));
  auto p_store = l_store.get_physical_store();
  auto acc     = p_store.read_accessor<int64_t, 1>();
  EXPECT_EQ(acc[2], 43);
}

TEST_F(InlineMap, Future) { test_inline_map_future(); }

TEST_F(InlineMap, RegionAndSlice) { test_inline_map_region_and_slice(); }

TEST_F(InlineMap, WithTask)
{
  register_tasks();
  test_inline_map_and_task();
}

}  // namespace inline_map
