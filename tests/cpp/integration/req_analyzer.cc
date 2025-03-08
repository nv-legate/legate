/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace req_analyzer {

// NOLINTBEGIN(readability-magic-numbers)

namespace {

struct Tester : public legate::LegateTask<Tester> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto inputs  = context.inputs();
    auto outputs = context.outputs();
    for (auto& input : inputs) {
      (void)input.data().read_accessor<std::int64_t, 2>();
    }
    for (auto& output : outputs) {
      (void)output.data().read_accessor<std::int64_t, 2>();
      (void)output.data().write_accessor<std::int64_t, 2>();
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_req_analyzer";
  static void registration_callback(legate::Library library) { Tester::register_variants(library); }
};

class ReqAnalyzer : public RegisterOnceFixture<Config> {};

void test_inout_store()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto store1 = runtime->create_store(legate::Shape{10, 5}, legate::int64());
  auto store2 = runtime->create_store(legate::Shape{10, 5}, legate::int64());
  runtime->issue_fill(store1, legate::Scalar{std::int64_t{0}});
  runtime->issue_fill(store2, legate::Scalar{std::int64_t{0}});

  auto task  = runtime->create_task(context, Tester::TASK_CONFIG.task_id());
  auto part1 = task.add_input(store1);
  auto part2 = task.add_input(store2);
  task.add_output(store1);
  task.add_constraint(legate::align(part1, part2));
  runtime->submit(std::move(task));
}

void test_isomorphic_transformed_stores()
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);

  auto store = runtime->create_store(legate::Shape{10}, legate::int64());
  runtime->issue_fill(store, legate::Scalar{std::int64_t{0}});

  // Create aliased stores that are semantically equivalent
  auto promoted1 = store.promote(1, 5);
  auto promoted2 = store.promote(1, 5);
  auto task      = runtime->create_task(context, Tester::TASK_CONFIG.task_id());
  task.add_input(promoted1);
  task.add_output(promoted2);
  runtime->submit(std::move(task));
}

}  // namespace

TEST_F(ReqAnalyzer, InoutStore) { test_inout_store(); }

TEST_F(ReqAnalyzer, IsomorphicTransformedStores) { test_isomorphic_transformed_stores(); }

// NOLINTEND(readability-magic-numbers)

}  // namespace req_analyzer
