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
  static void cpu_variant(legate::TaskContext context) {}
};

void prepare()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  Tester::register_variants(context);
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

TEST(ReqAnalyzer, IsomorphicTransformedStores)
{
  legate::Core::perform_registration<prepare>();
  test_isomorphic_transformed_stores();
}

}  // namespace req_analyzer
