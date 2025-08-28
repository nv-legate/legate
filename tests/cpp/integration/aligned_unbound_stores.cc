/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace aligned_unbound_stores_test {

struct Producer : public legate::LegateTask<Producer> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto outputs = context.outputs();
    for (auto&& output : outputs) {
      output.data().bind_empty_data();
      if (output.nullable()) {
        output.null_mask().bind_empty_data();
      }
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_unbound_nullable_array";

  static void registration_callback(legate::Library library)
  {
    Producer::register_variants(library);
  }
};

class AlignedUnboundStores : public RegisterOnceFixture<Config> {};

TEST_F(AlignedUnboundStores, Standalone)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto store1 = runtime->create_store(legate::int32());
  auto store2 = runtime->create_store(legate::int64());

  {
    auto task  = runtime->create_task(library, Producer::TASK_CONFIG.task_id());
    auto part1 = task.add_output(store1);
    auto part2 = task.add_output(store2);
    task.add_constraint(legate::align(part1, part2));
    runtime->submit(std::move(task));
  }
  EXPECT_EQ(store1.shape().extents(), legate::tuple<std::uint64_t>{0});
}

TEST_F(AlignedUnboundStores, ViaNullableArray)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto arr = runtime->create_array(legate::int32(), 1, true /*nullable*/);

  {
    auto task = runtime->create_task(library, Producer::TASK_CONFIG.task_id());
    task.add_output(arr);
    runtime->submit(std::move(task));
  }
  EXPECT_EQ(arr.shape().extents(), legate::tuple<std::uint64_t>{0});
}

}  // namespace aligned_unbound_stores_test
