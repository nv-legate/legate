/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/timing/timing.h>

#include <legate.h>

#include <legate/operation/detail/timing.h>
#include <legate/runtime/detail/runtime.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace {

// NOLINTBEGIN(readability-magic-numbers)

void hello_cpu_variant(legate::TaskContext& context)
{
  auto output = context.output(0);
  auto shape  = output.shape<2>();
  if (shape.empty()) {
    return;
  }
  auto acc = output.write_accessor<std::int64_t, 2>(shape);

  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    acc[*it] = (*it)[0] + ((*it)[1] * 1000);
  }
}

struct HelloTask : legate::LegateTask<HelloTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context) { hello_cpu_variant(context); };
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_timing";

  static void registration_callback(legate::Library library)
  {
    HelloTask::register_variants(library);
  }
};

class Timing : public RegisterOnceFixture<Config> {};

void test_hello_task()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto store   = runtime->create_store(legate::Shape{1000, 1000}, legate::int64());
  auto task    = runtime->create_task(library, HelloTask::TASK_CONFIG.task_id());
  auto part    = task.declare_partition();

  task.add_output(store, part);
  runtime->submit(std::move(task));
}

TEST_F(Timing, MeasureMicroseconds)
{
  auto t1 = legate::timing::measure_microseconds();
  test_hello_task();
  auto t2 = legate::timing::measure_microseconds();
  EXPECT_GT(t2.value(), t1.value());
}

TEST_F(Timing, MeasureNanoseconds)
{
  auto t1 = legate::timing::measure_nanoseconds();
  test_hello_task();
  auto t2 = legate::timing::measure_nanoseconds();
  EXPECT_GT(t2.value(), t1.value());
}

TEST_F(Timing, LaunchWithStrategyDelegatesToLaunch)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{}, legate::int64(), /*optimize_scalar=*/true);
  auto timing  = legate::detail::Timing{
    runtime->impl()->new_op_id(), legate::detail::Timing::Precision::MICRO, store.impl()};

  ASSERT_EQ(timing.kind(), legate::detail::Operation::Kind::TIMING);

  timing.launch(static_cast<legate::detail::Strategy*>(nullptr));

  auto future = store.impl()->get_future();

  ASSERT_TRUE(future.exists());
  ASSERT_GE(future.get_result<std::int64_t>(), 0);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace
