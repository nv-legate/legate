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

#include "timing/timing.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace {

static constexpr const char* library_name = "test_timing";

void hello_cpu_variant(legate::TaskContext& context)
{
  auto output = context.output(0).data();
  auto shape  = output.shape<2>();
  if (shape.empty()) {
    return;
  }
  auto acc = output.write_accessor<int64_t, 2>(shape);

  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    acc[*it] = (*it)[0] + (*it)[1] * 1000;
  }
}

struct HelloTask : legate::LegateTask<HelloTask> {
  static constexpr std::int32_t TASK_ID = 0;
  static void cpu_variant(legate::TaskContext context) { hello_cpu_variant(context); };
};

struct Timing : DefaultFixture {
  void SetUp() override
  {
    DefaultFixture::SetUp();
    auto runtime = legate::Runtime::get_runtime();
    auto library = runtime->create_library(library_name);
    HelloTask::register_variants(library);
  }
};

void test_hello_task()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(library_name);
  auto store   = runtime->create_store(legate::Shape{1000, 1000}, legate::int64());
  auto task    = runtime->create_task(library, HelloTask::TASK_ID);
  auto part    = task.declare_partition();

  task.add_output(store, part);
  runtime->submit(std::move(task));
}

TEST_F(Timing, measure_microseconds)
{
  auto t1 = legate::timing::measure_microseconds();
  test_hello_task();
  auto t2 = legate::timing::measure_microseconds();
  EXPECT_GT(t2.value(), t1.value());
}

TEST_F(Timing, measure_nanoseconds)
{
  auto t1 = legate::timing::measure_nanoseconds();
  test_hello_task();
  auto t2 = legate::timing::measure_nanoseconds();
  EXPECT_GT(t2.value(), t1.value());
}

}  // namespace
