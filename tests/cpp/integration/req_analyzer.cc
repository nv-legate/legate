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

namespace req_analyzer {

static const char* library_name = "test_req_analyzer";

enum TaskIDs {
  TESTER = 0,
};

struct Tester : public legate::LegateTask<Tester> {
  static const int32_t TASK_ID = TESTER;
  static void cpu_variant(legate::TaskContext context)
  {
    auto inputs  = context.inputs();
    auto outputs = context.outputs();
    for (auto& input : inputs) { input.data().read_accessor<int64_t, 2>(); }
    for (auto& output : outputs) {
      output.data().read_accessor<int64_t, 2>();
      output.data().write_accessor<int64_t, 2>();
    }
  }
};

void prepare()
{
  static bool prepared = false;
  if (prepared) { return; }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  Tester::register_variants(context);
}

void test_inout_store()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto store1 = runtime->create_store({10, 5}, legate::int64());
  auto store2 = runtime->create_store({10, 5}, legate::int64());
  runtime->issue_fill(store1, legate::Scalar(int64_t{0}));
  runtime->issue_fill(store2, legate::Scalar(int64_t{0}));

  auto task  = runtime->create_task(context, TESTER);
  auto part1 = task.add_input(store1);
  auto part2 = task.add_input(store2);
  task.add_output(store1);
  task.add_constraint(legate::align(part1, part2));
  runtime->submit(std::move(task));
}

void test_isomorphic_transformed_stores()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);

  auto store = runtime->create_store({10}, legate::int64());
  runtime->issue_fill(store, legate::Scalar(int64_t{0}));

  // Create aliased stores that are semantically equivalent
  auto promoted1 = store.promote(1, 5);
  auto promoted2 = store.promote(1, 5);
  auto task      = runtime->create_task(context, TESTER);
  task.add_input(promoted1);
  task.add_output(promoted2);
  runtime->submit(std::move(task));
}

TEST(ReqAnalyzer, InoutStore)
{
  prepare();
  test_inout_store();
}

TEST(ReqAnalyzer, IsomorphicTransformedStores)
{
  prepare();
  test_isomorphic_transformed_stores();
}

}  // namespace req_analyzer
