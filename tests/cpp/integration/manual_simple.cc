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

namespace manualsimple {

void test_auto_task(legate::Library library, legate::LogicalStore store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, task::simple::HELLO);
  auto part    = task.declare_partition();
  task.add_output(store, part);
  runtime->submit(std::move(task));
}

void test_manual_task(legate::Library library, legate::LogicalStore store)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, task::simple::HELLO, {3, 3});
  auto part    = store.partition_by_tiling({2, 2});
  task.add_output(part);
  runtime->submit(std::move(task));
}

void print_store(legate::LogicalStore store)
{
  auto runtime = legate::Runtime::get_runtime();
  static_cast<void>(runtime);
  auto p_store = store.get_physical_store();
  auto acc     = p_store.read_accessor<int64_t, 2>();
  auto shape   = p_store.shape<2>();
  std::stringstream ss;
  for (legate::PointInRectIterator<2> it(shape); it.valid(); ++it) ss << acc[*it] << " ";
  task::simple::logger.print() << ss.str();
}

TEST(Integration, ManualSimple)
{
  legate::Core::perform_registration<task::simple::register_tasks>();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(task::simple::library_name);

  auto store = runtime->create_store({5, 5}, legate::int64());
  test_auto_task(library, store);
  print_store(store);
  test_manual_task(library, store);
  print_store(store);
}

}  // namespace manualsimple
