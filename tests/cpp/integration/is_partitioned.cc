/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace is_partitioned {

namespace {

constexpr std::uint64_t EXTENT    = 42;
constexpr std::uint64_t NUM_TASKS = 2;

struct Tester : public legate::LegateTask<Tester> {
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  static void cpu_variant(legate::TaskContext context)
  {
    const auto output1 = context.output(0).data();
    const auto output2 = context.output(1).data();
    const auto output3 = context.output(2).data();

    if (!context.is_single_task()) {
      EXPECT_TRUE(output1.is_partitioned());
      EXPECT_TRUE(output2.is_partitioned());
    }
    EXPECT_FALSE(output3.is_partitioned());

    output2.bind_empty_data();
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_is_partitioned";
  static void registration_callback(legate::Library library) { Tester::register_variants(library); }
};

class IsPartitioned : public RegisterOnceFixture<Config> {};

}  // namespace

TEST_F(IsPartitioned, Auto)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto store1 = runtime->create_store(legate::Shape{EXTENT}, legate::int64());
  auto store2 = runtime->create_store(legate::int64());
  auto store3 = runtime->create_store(legate::Shape{EXTENT}, legate::int64());

  auto task = runtime->create_task(library, Tester::TASK_ID);
  task.add_output(store1);
  task.add_output(store2);
  auto part = task.add_output(store3);
  task.add_constraint(legate::broadcast(part));
  runtime->submit(std::move(task));
}

TEST_F(IsPartitioned, Manual)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto store1 = runtime->create_store(legate::Shape{EXTENT}, legate::int64());
  auto store2 = runtime->create_store(legate::int64());
  auto store3 = runtime->create_store(legate::Shape{EXTENT}, legate::int64());

  auto task = runtime->create_task(library, Tester::TASK_ID, {NUM_TASKS});
  task.add_output(store1.partition_by_tiling({EXTENT / NUM_TASKS}));
  task.add_output(store2);
  task.add_output(store3);
  runtime->submit(std::move(task));
}

}  // namespace is_partitioned
