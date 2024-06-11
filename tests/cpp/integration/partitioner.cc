/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace partitioner_test {

// NOLINTBEGIN(readability-magic-numbers)

struct Initializer : public legate::LegateTask<Initializer> {
  static const std::int32_t TASK_ID = 0;
  static void cpu_variant(legate::TaskContext /*context*/) {}
};

struct Checker : public legate::LegateTask<Checker> {
  static const std::int32_t TASK_ID = 1;
  static void cpu_variant(legate::TaskContext context) { EXPECT_FALSE(context.is_single_task()); }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_partitioner";
  static void registration_callback(legate::Library library)
  {
    Initializer::register_variants(library);
    Checker::register_variants(library);
  }
};

class PartitionerTest : public RegisterOnceFixture<Config> {};

TEST_F(PartitionerTest, FavorPartitionedStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto store1 = runtime->create_store(legate::Shape{10, 10}, legate::int64());
  auto store2 = runtime->create_store(legate::Shape{10, 10}, legate::int64());

  // Initialize store1 sequentially
  {
    auto task = runtime->create_task(library, Initializer::TASK_ID);
    auto part = task.add_output(store1);
    task.add_constraint(legate::broadcast(part));
    runtime->submit(std::move(task));
  }

  // Initialize store2 with parallel tasks
  {
    auto part = store2.partition_by_tiling({5, 5});
    auto task = runtime->create_task(library, Initializer::TASK_ID, part.color_shape());
    task.add_output(part);
    runtime->submit(std::move(task));
  }

  // Because the partitioner favors partitioned stores over non-partitioned ones, the checker task
  // should always get parallelized
  {
    auto task  = runtime->create_task(library, Checker::TASK_ID);
    auto part1 = task.add_input(store1);
    auto part2 = task.add_input(store2);
    task.add_constraint(legate::align(part1, part2));
    runtime->submit(std::move(task));
  }
}

// NOLINTEND(readability-magic-numbers)

}  // namespace partitioner_test
