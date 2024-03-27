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

#include "legate.h"
#include "tasks/task_simple.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace auto_task_test {

using AutoTask = DefaultFixture;

TEST_F(AutoTask, Invalid)
{
  task::simple::register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::simple::library_name);

  auto unbound_array = runtime->create_array(legate::int64(), 1);

  auto task = runtime->create_task(library, task::simple::HELLO);

  // Unbound arrays cannot be used for inputs or reductions
  EXPECT_THROW(task.add_input(unbound_array), std::invalid_argument);
  EXPECT_THROW(task.add_reduction(unbound_array, legate::ReductionOpKind::ADD),
               std::invalid_argument);
}

}  // namespace auto_task_test
