/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/experimental/trace.h>
#include <legate/utilities/detail/env.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace tracing_test {

// NOLINTBEGIN(readability-magic-numbers)
namespace {

struct DummyTask : public legate::LegateTask<DummyTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

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
    auto task = runtime->create_task(library, DummyTask::TASK_CONFIG.task_id());
    task.add_input(array);
    runtime->submit(std::move(task));
  }
  {
    auto task = runtime->create_task(library, DummyTask::TASK_CONFIG.task_id());
    task.add_input(array);
    task.add_output(array);
    runtime->submit(std::move(task));
  }
}

constexpr std::uint32_t NUM_ITER = 10;
constexpr std::uint32_t TRACE_ID = 42;

}  // namespace

// TODO(wonchanl)
// Disabling the tracing test, as tracing doesn't support task variants that have no
// registration-time return sizes.
TEST_F(Tracing, DISABLED_RAII)
{
  // TODO(jfaibussowit)
  //
  // LEGION ERROR: Illegal runtime remapping in trace 42 inside of task Legate Core Toplevel
  // Task (UID 1). Traces must perfectly manage their physical mappings with no runtime
  // help. (from file
  // legate/arch-darwin-debug/cmake_build/_deps/legion-src/runtime/legion/legion_context.cc:924)
  if (legate::detail::experimental::LEGATE_INLINE_TASK_LAUNCH.get(/* default_value = */ false)) {
    return;
  }
  auto runtime = legate::Runtime::get_runtime();
  auto array   = runtime->create_array(legate::Shape{10}, legate::int64());
  launch_tasks(array);
  for (std::uint32_t idx = 0; idx < NUM_ITER; ++idx) {
    const legate::experimental::Trace trace{TRACE_ID};

    launch_tasks(array);
  }
}

// TODO(wonchanl)
// Disabling the tracing test, as tracing doesn't support task variants that have no
// registration-time return sizes.
TEST_F(Tracing, DISABLED_BeginEnd)
{
  // TODO(jfaibussowit)
  //
  // LEGION ERROR: Illegal runtime remapping in trace 42 inside of task Legate Core Toplevel
  // Task (UID 1). Traces must perfectly manage their physical mappings with no runtime
  // help. (from file
  // legate/arch-darwin-debug/cmake_build/_deps/legion-src/runtime/legion/legion_context.cc:924)
  if (legate::detail::experimental::LEGATE_INLINE_TASK_LAUNCH.get(/* default_value = */ false)) {
    return;
  }

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
