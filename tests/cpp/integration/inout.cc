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

#include <gtest/gtest.h>

#include "legate.h"
#include "tasks/task_simple.h"

namespace inout {

void test_inout()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(task::simple::library_name);

  auto store = runtime->create_store({10, 10}, legate::int64());
  runtime->issue_fill(store, legate::Scalar(int64_t(0)));

  auto task = runtime->create_task(context, task::simple::HELLO);
  task.add_input(store, task.find_or_declare_partition(store));
  task.add_output(store, task.find_or_declare_partition(store));
  runtime->submit(std::move(task));
}

TEST(Integration, InOut)
{
  legate::Core::perform_registration<task::simple::register_tasks>();
  test_inout();
}

}  // namespace inout
