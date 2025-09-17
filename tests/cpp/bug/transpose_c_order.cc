/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_transpose_c_order {

class Tester : public legate::LegateTask<Tester> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto store     = context.output(0).data();
    auto store_acc = store.read_accessor<std::int64_t, 3>();
    EXPECT_EQ(&(store_acc[{0, 0, 1}]) - &(store_acc[{0, 0, 0}]), 1);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_transpose_c_order";

  static void registration_callback(legate::Library library) { Tester::register_variants(library); }
};

class TransposeCOrder : public RegisterOnceFixture<Config> {};

TEST_F(TransposeCOrder, Test)
{
  constexpr int X = 3;
  constexpr int Y = 4;
  constexpr int Z = 5;
  auto runtime    = legate::Runtime::get_runtime();
  auto shape      = legate::Shape{X, Y, Z};
  auto store      = runtime->create_store(shape, legate::int64());

  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(library, Tester::TASK_CONFIG.task_id(), {1});
  task.add_output(store.transpose({2, 0, 1}));
  runtime->submit(std::move(task));
}

}  // namespace test_transpose_c_order
