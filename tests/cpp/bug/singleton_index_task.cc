/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace singleton_index_task_test {

using SingletonIndexTask = DefaultFixture;

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr std::string_view LIBRARY_NAME = "test_singleton_index_task";

}  // namespace

struct Checker : public legate::LegateTask<Checker> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    EXPECT_EQ(context.num_communicators(), 0);
    EXPECT_TRUE(context.communicators().empty());
  }
};

TEST_F(SingletonIndexTask, Bug1)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(LIBRARY_NAME);
  Checker::register_variants(library);

  auto store = runtime->create_store(legate::Shape{10, 1}, legate::int64());
  runtime->issue_fill(store, legate::Scalar{int64_t{42}});

  auto task  = runtime->create_task(library, Checker::TASK_CONFIG.task_id());
  auto part1 = task.add_output(store.project(0, 0));
  auto part2 = task.add_output(runtime->create_store(legate::Scalar{42}));
  task.add_constraint(legate::align(part1, part2));
  task.add_communicator("cpu");
  runtime->submit(std::move(task));
}

// NOLINTEND(readability-magic-numbers)

}  // namespace singleton_index_task_test
