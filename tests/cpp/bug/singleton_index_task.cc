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
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace singleton_index_task_test {

using SingletonIndexTask = DefaultFixture;

constexpr const char* library_name = "test_singleton_index_task";

struct Checker : public legate::LegateTask<Checker> {
  static const std::int32_t TASK_ID = 0;
  static void cpu_variant(legate::TaskContext context)
  {
    EXPECT_EQ(context.num_communicators(), 0);
    EXPECT_TRUE(context.communicators().empty());
  }
};

TEST_F(SingletonIndexTask, Bug1)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  Checker::register_variants(library);

  auto store = runtime->create_store(legate::Shape{10, 1}, legate::int64());
  runtime->issue_fill(store, legate::Scalar{int64_t{42}});

  auto task  = runtime->create_task(library, Checker::TASK_ID);
  auto part1 = task.add_output(store.project(0, 0));
  auto part2 = task.add_output(runtime->create_store(legate::Scalar{42}));
  task.add_constraint(legate::align(part1, part2));
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
}

}  // namespace singleton_index_task_test
