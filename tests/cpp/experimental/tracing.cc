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

#include "core/experimental/trace.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace tracing_test {

using Tracing = DefaultFixture;

struct DummyTask : public legate::LegateTask<DummyTask> {
  static constexpr std::int32_t TASK_ID = 0;
  static void cpu_variant(legate::TaskContext /*context*/) {}
};

TEST_F(Tracing, Test)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library("tracing_test");
  DummyTask::register_variants(library);

  auto store = runtime->create_array(legate::Shape{10}, legate::int64());

  auto launch_tasks = [&runtime, &library, &store]() {
    runtime->issue_fill(store, legate::Scalar{std::int64_t{123}});
    {
      auto task = runtime->create_task(library, DummyTask::TASK_ID);
      task.add_input(store);
      runtime->submit(std::move(task));
    }
    {
      auto task = runtime->create_task(library, DummyTask::TASK_ID);
      task.add_input(store);
      task.add_output(store);
      runtime->submit(std::move(task));
    }
  };

  constexpr std::uint32_t NUM_ITER = 10;
  launch_tasks();
  for (std::uint32_t idx = 0; idx < NUM_ITER; ++idx) {
    legate::experimental::Trace trace{42};
    launch_tasks();
  }
}

}  // namespace tracing_test
