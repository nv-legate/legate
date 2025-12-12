/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace alias_via_promote_test {

using AliasViaPromote = DefaultFixture;

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr std::string_view LIBRARY_NAME = "test_alias_via_promote";

}  // namespace

struct Checker : public legate::LegateTask<Checker> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

TEST_F(AliasViaPromote, Bug1)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(LIBRARY_NAME);
  Checker::register_variants(library);

  auto store = runtime->create_store(legate::Shape{2}, legate::int64());
  runtime->issue_fill(store, legate::Scalar{int64_t{42}});

  auto task  = runtime->create_task(library, Checker::TASK_CONFIG.task_id());
  auto part1 = task.add_output(store.promote(/*extra_dim=*/1, /*dim_size=*/100));
  task.add_constraint(legate::broadcast(part1, {0}));
  runtime->submit(std::move(task));
}

// NOLINTEND(readability-magic-numbers)

}  // namespace alias_via_promote_test
