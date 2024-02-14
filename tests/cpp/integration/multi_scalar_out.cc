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

namespace multiscalarout {

using Integration = DefaultFixture;

void test_writer_auto(legate::Library library,
                      legate::LogicalStore scalar1,
                      legate::LogicalStore scalar2)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, task::simple::WRITER);
  task.add_output(scalar1);
  task.add_output(scalar2);
  runtime->submit(std::move(task));
}

void test_reducer_auto(legate::Library library,
                       legate::LogicalStore scalar1,
                       legate::LogicalStore scalar2,
                       legate::LogicalStore store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, task::simple::REDUCER);
  task.add_reduction(scalar1, legate::ReductionOpKind::ADD);
  task.add_reduction(scalar2, legate::ReductionOpKind::MUL);
  task.add_input(store);
  runtime->submit(std::move(task));
}

void test_reducer_manual(legate::Library library,
                         legate::LogicalStore scalar1,
                         legate::LogicalStore scalar2,
                         legate::LogicalStore store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, task::simple::REDUCER, {2});
  task.add_reduction(scalar1, legate::ReductionOpKind::ADD);
  task.add_reduction(scalar2, legate::ReductionOpKind::MUL);
  task.add_input(store.partition_by_tiling({3}));
  runtime->submit(std::move(task));
}

void validate_stores(legate::LogicalStore scalar1,
                     legate::LogicalStore scalar2,
                     std::int32_t to_match1,
                     std::int64_t to_match2)
{
  auto runtime = legate::Runtime::get_runtime();
  static_cast<void>(runtime);
  auto p_scalar1 = scalar1.get_physical_store();
  auto p_scalar2 = scalar2.get_physical_store();
  auto acc1      = p_scalar1.read_accessor<int32_t, 2>();
  auto acc2      = p_scalar2.read_accessor<int64_t, 3>();
  auto v1        = acc1[{0, 0}];
  auto v2        = acc2[{0, 0, 0}];
  EXPECT_EQ(v1, to_match1);
  EXPECT_EQ(v2, to_match2);
}

TEST_F(Integration, MultiScalarOut)
{
  task::simple::register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::simple::library_name);

  auto scalar1 = runtime->create_store(legate::Shape{1, 1}, legate::int32(), true);
  auto scalar2 = runtime->create_store(legate::Shape{1, 1, 1}, legate::int64(), true);
  auto store   = runtime->create_store(legate::Shape{5}, legate::int64());
  runtime->issue_fill(store, legate::Scalar(int64_t(0)));

  test_writer_auto(library, scalar1, scalar2);
  validate_stores(scalar1, scalar2, 10, 20);
  test_reducer_auto(library, scalar1, scalar2, store);
  validate_stores(scalar1, scalar2, 60, 640);
  test_reducer_manual(library, scalar1, scalar2, store);
  validate_stores(scalar1, scalar2, 110, 20480);
}

}  // namespace multiscalarout
