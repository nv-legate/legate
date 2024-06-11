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

#include "core/experimental/trace.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace tracing_test {

// NOLINTBEGIN(readability-magic-numbers)

struct DummyTask : public legate::LegateTask<DummyTask> {
  static constexpr std::int32_t TASK_ID = 0;
  static void cpu_variant(legate::TaskContext /*context*/) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "tracing_test";
  static void registration_callback(legate::Library library)
  {
    DummyTask::register_variants(library);
  }
};

class Tracing : public RegisterOnceFixture<Config> {};

void launch_tasks(legate::LogicalArray& array)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  runtime->issue_fill(array, legate::Scalar{std::int64_t{123}});
  {
    auto task = runtime->create_task(library, DummyTask::TASK_ID);
    task.add_input(array);
    runtime->submit(std::move(task));
  }
  {
    auto task = runtime->create_task(library, DummyTask::TASK_ID);
    task.add_input(array);
    task.add_output(array);
    runtime->submit(std::move(task));
  }
}

constexpr std::uint32_t NUM_ITER = 10;
constexpr std::uint32_t TRACE_ID = 42;

TEST_F(Tracing, RAII)
{
  auto runtime = legate::Runtime::get_runtime();
  auto array   = runtime->create_array(legate::Shape{10}, legate::int64());
  launch_tasks(array);
  for (std::uint32_t idx = 0; idx < NUM_ITER; ++idx) {
    const legate::experimental::Trace trace{TRACE_ID};

    launch_tasks(array);
  }
}

TEST_F(Tracing, BeginEnd)
{
  auto runtime = legate::Runtime::get_runtime();
  auto array   = runtime->create_array(legate::Shape{10}, legate::int64());
  launch_tasks(array);
  for (std::uint32_t idx = 0; idx < NUM_ITER; ++idx) {
    legate::experimental::Trace::begin_trace(TRACE_ID);
    launch_tasks(array);
    legate::experimental::Trace::end_trace(TRACE_ID);
  }
}

// NOLINTEND(readability-magic-numbers)

}  // namespace tracing_test
