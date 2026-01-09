/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <integration/tasks/task_simple.h>
#include <utilities/utilities.h>

namespace manual_task_test {

// NOLINTBEGIN(readability-magic-numbers)

namespace {

using ManualTask = DefaultFixture;

void test_auto_task(legate::Library library, const legate::LogicalStore& store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, task::simple::HelloTask::TASK_CONFIG.task_id());
  auto part    = task.declare_partition();
  task.add_output(store, part);
  runtime->submit(std::move(task));
}

void test_manual_task(legate::Library library, const legate::LogicalStore& store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(
    library, task::simple::HelloTask::TASK_CONFIG.task_id(), legate::tuple<std::uint64_t>{3, 3});
  auto part = store.partition_by_tiling({2, 2});
  task.add_output(part);
  runtime->submit(std::move(task));
}

void validate_store(const legate::LogicalStore& store)
{
  auto runtime = legate::Runtime::get_runtime();
  static_cast<void>(runtime);
  auto p_store = store.get_physical_store();
  auto acc     = p_store.read_accessor<std::int64_t, 2>();
  auto shape   = p_store.shape<2>();
  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    auto p = *it;
    EXPECT_EQ(acc[p], p[0] + (p[1] * 1000));
  }
}

}  // namespace

TEST_F(ManualTask, Simple)
{
  task::simple::register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::simple::LIBRARY_NAME);

  auto store = runtime->create_store(legate::Shape{5, 5}, legate::int64());
  test_auto_task(library, store);
  validate_store(store);
  test_manual_task(library, store);
  validate_store(store);
}

TEST_F(ManualTask, Invalid)
{
  task::simple::register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::simple::LIBRARY_NAME);

  auto scalar_store  = runtime->create_store(legate::Scalar{1});
  auto unbound_store = runtime->create_store(legate::int64(), 1);

  auto task = runtime->create_task(
    library, task::simple::HelloTask::TASK_CONFIG.task_id(), legate::tuple<std::uint64_t>{3, 3});
  // Unbound stores cannot be used for inputs or reductions
  EXPECT_THROW(task.add_input(unbound_store), std::invalid_argument);
  EXPECT_THROW(task.add_reduction(unbound_store, legate::ReductionOpKind::ADD),
               std::invalid_argument);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace manual_task_test
