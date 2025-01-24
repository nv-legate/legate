/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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

  auto unbound_array = runtime->create_array(legate::int64(), 1);

  auto task = runtime->create_task(library, task::simple::HelloTask::TASK_ID);

  // Unbound arrays cannot be used for inputs or reductions
  EXPECT_THROW(task.add_input(unbound_array), std::invalid_argument);
  EXPECT_THROW(task.add_reduction(unbound_array, legate::ReductionOpKind::ADD),
               std::invalid_argument);
}

TEST_F(AutoTask, InvalidListArray)
{
  task::simple::register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::simple::LIBRARY_NAME);

  auto type       = legate::list_type(legate::int64());
  auto descriptor = runtime->create_array(legate::Shape{2}, legate::rect_type(1), true);
  auto vardata    = runtime->create_array(legate::Shape{3}, legate::int64());
  auto list_array = runtime->create_list_array(descriptor, vardata, type);

  auto task = runtime->create_task(library, task::simple::HelloTask::TASK_ID);

  // List array cannot be used for reductions
  EXPECT_THROW(task.add_reduction(list_array, legate::ReductionOpKind::ADD), std::invalid_argument);
}

}  // namespace auto_task_test
