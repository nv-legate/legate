/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <integration/tasks/task_simple.h>
#include <utilities/utilities.h>

namespace physical_task_test {

namespace {

using PhysicalTask = DefaultFixture;

// Test constants
constexpr std::int64_t TEST_SCALAR_VALUE = 42;
constexpr std::uint64_t TEST_ARRAY_SIZE  = 5;
constexpr std::uint64_t SMALL_ARRAY_SIZE = 3;

void test_physical_task_basic_apis(legate::Library library, const legate::LogicalStore& store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task =
    runtime->create_physical_task(library, task::simple::HelloTask::TASK_CONFIG.task_id());

  // Create a PhysicalArray for testing
  auto logical_array  = legate::LogicalArray{store};
  auto physical_array = logical_array.get_physical_array();

  // Test that PhysicalTask methods work correctly without throwing exceptions
  EXPECT_NO_THROW(task.add_input(physical_array));
  EXPECT_NO_THROW(task.add_output(physical_array));
  EXPECT_NO_THROW(
    task.add_reduction(physical_array, static_cast<std::int32_t>(legate::ReductionOpKind::ADD)));

  // Note: Partition-related methods are not supported for PhysicalTask as it bypasses partitioning
}

void test_physical_task_other_apis(legate::Library library)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task =
    runtime->create_physical_task(library, task::simple::HelloTask::TASK_CONFIG.task_id());

  // Test that these APIs work (they don't depend on PhysicalArray)
  EXPECT_NO_THROW(task.add_scalar_arg(legate::Scalar{TEST_SCALAR_VALUE}));

  // Test boolean setters
  EXPECT_NO_THROW(task.set_concurrent(true));
  EXPECT_NO_THROW(task.set_side_effect(false));
  EXPECT_NO_THROW(task.throws_exception(true));

  // Note: Partition-related methods (declare_partition, find_or_declare_partition)
  // and communicator methods are not supported for PhysicalTask
}

}  // namespace

TEST_F(PhysicalTask, BasicAPIs)
{
  task::simple::register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::simple::LIBRARY_NAME);

  auto store =
    runtime->create_store(legate::Shape{TEST_ARRAY_SIZE, TEST_ARRAY_SIZE}, legate::int64());

  // Test that PhysicalTask APIs work correctly
  EXPECT_NO_THROW(test_physical_task_basic_apis(library, store));
}

TEST_F(PhysicalTask, OtherAPIs)
{
  task::simple::register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::simple::LIBRARY_NAME);

  // Test that non-PhysicalArray APIs work correctly
  EXPECT_NO_THROW(test_physical_task_other_apis(library));
}

TEST_F(PhysicalTask, ErrorMessages)
{
  task::simple::register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::simple::LIBRARY_NAME);
  auto task =
    runtime->create_physical_task(library, task::simple::HelloTask::TASK_CONFIG.task_id());

  auto store =
    runtime->create_store(legate::Shape{SMALL_ARRAY_SIZE, SMALL_ARRAY_SIZE}, legate::int64());
  auto logical_array  = legate::LogicalArray{store};
  auto physical_array = logical_array.get_physical_array();

  // Test that PhysicalTask methods work correctly
  EXPECT_NO_THROW(task.add_input(physical_array));
  EXPECT_NO_THROW(task.add_output(physical_array));
  EXPECT_NO_THROW(
    task.add_reduction(physical_array, static_cast<std::int32_t>(legate::ReductionOpKind::ADD)));
}

}  // namespace physical_task_test
