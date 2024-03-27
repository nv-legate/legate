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

namespace alias_via_promote_test {

using AliasViaPromote = DefaultFixture;

constexpr const char* library_name = "test_alias_via_promote";

struct Checker : public legate::LegateTask<Checker> {
  static const std::int32_t TASK_ID = 0;
  static void cpu_variant(legate::TaskContext /*context*/) {}
};

TEST_F(AliasViaPromote, Bug1)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  Checker::register_variants(library);

  auto store = runtime->create_store(legate::Shape{2}, legate::int64());
  runtime->issue_fill(store, legate::Scalar{int64_t{42}});

  auto task  = runtime->create_task(library, Checker::TASK_ID);
  auto part1 = task.add_output(store.promote(1, 100));
  task.add_constraint(legate::broadcast(part1, {0}));
  runtime->submit(std::move(task));
}

}  // namespace alias_via_promote_test
