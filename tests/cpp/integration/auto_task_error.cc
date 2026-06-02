/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <integration/tasks/task_simple.h>
#include <utilities/utilities.h>

namespace auto_task_test {

using AutoTask = DefaultFixture;

TEST_F(AutoTask, InvalidUnboundArray)
{
  task::simple::register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::simple::LIBRARY_NAME);

  auto unbound_store = runtime->create_store(legate::int64(), /*dim=*/1);

  auto task = runtime->create_task(library, task::simple::HelloTask::TASK_CONFIG.task_id());

  // Unbound arrays cannot be used for inputs or reductions
  EXPECT_THROW(task.add_input(unbound_store), std::invalid_argument);
  EXPECT_THROW(task.add_reduction(unbound_store, legate::ReductionOpKind::ADD),
               std::invalid_argument);
}

TEST_F(AutoTask, InvalidPartition)
{
  task::simple::register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::simple::LIBRARY_NAME);

  auto store_input1 = runtime->create_store(legate::Shape{3}, legate::int32());
  auto store_input2 = runtime->create_store(legate::Shape{3}, legate::int32());

  auto task = runtime->create_task(library, task::simple::HelloTask::TASK_CONFIG.task_id());
  auto part = task.declare_partition();

  task.add_input(store_input1, part);

  // Use the same partition for two inputs, should throw std::invalid_argument
  EXPECT_THROW(task.add_input(store_input2, part), std::invalid_argument);
}

}  // namespace auto_task_test
