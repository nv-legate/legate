/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <integration/tasks/task_simple.h>
#include <utilities/utilities.h>

namespace test_physical_task {

constexpr std::int64_t TEST_DIMENSION_SIZE = 10;

class PhysicalTaskUnit : public DefaultFixture {};

TEST_F(PhysicalTaskUnit, CreatePhysicalTask)
{
  task::simple::register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::simple::LIBRARY_NAME);

  // Test that PhysicalTask creation succeeds using assertions
  ASSERT_NO_THROW({
    auto task =
      runtime->create_physical_task(library, task::simple::HelloTask::TASK_CONFIG.task_id());
    static_cast<void>(task);  // Suppress unused variable warning
  });
}

TEST_F(PhysicalTaskUnit, PhysicalTaskBasicMethods)
{
  task::simple::register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::simple::LIBRARY_NAME);

  auto task =
    runtime->create_physical_task(library, task::simple::HelloTask::TASK_CONFIG.task_id());

  // Create a store and convert to PhysicalArray for testing
  auto store =
    runtime->create_store(legate::Shape{TEST_DIMENSION_SIZE, TEST_DIMENSION_SIZE}, legate::int64());
  auto logical_array  = legate::LogicalArray{store};
  auto physical_array = logical_array.get_physical_array();

  // Test that basic PhysicalTask methods work correctly (no longer throw)
  EXPECT_NO_THROW(task.add_input(physical_array));
  EXPECT_NO_THROW(task.add_output(physical_array));
  EXPECT_NO_THROW(
    task.add_reduction(physical_array, static_cast<std::int32_t>(legate::ReductionOpKind::ADD)));

  // Test scalar methods work
  constexpr std::int32_t TEST_SCALAR_VALUE = 42;
  EXPECT_NO_THROW(task.add_scalar_arg(legate::Scalar{TEST_SCALAR_VALUE}));
  EXPECT_NO_THROW(task.set_concurrent(true));
  EXPECT_NO_THROW(task.set_side_effect(false));
  EXPECT_NO_THROW(task.throws_exception(true));
}

}  // namespace test_physical_task
