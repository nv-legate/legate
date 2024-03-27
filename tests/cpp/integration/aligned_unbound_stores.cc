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

namespace aligned_unbound_stores_test {

using AlignedUnboundStores = DefaultFixture;

constexpr const char* library_name = "test_unbound_nullable_array_test";

struct Producer : public legate::LegateTask<Producer> {
  static const std::int32_t TASK_ID = 0;
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

TEST_F(AlignedUnboundStores, Standalone)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  Producer::register_variants(library);

  auto store1 = runtime->create_store(legate::int32());
  auto store2 = runtime->create_store(legate::int64());

  {
    auto task  = runtime->create_task(library, Producer::TASK_ID);
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
  auto library = runtime->create_library(library_name);
  Producer::register_variants(library);

  auto arr = runtime->create_array(legate::int32(), 1, true /*nullable*/);

  {
    auto task = runtime->create_task(library, Producer::TASK_ID);
    task.add_output(arr);
    runtime->submit(std::move(task));
  }
  EXPECT_EQ(arr.shape().extents(), legate::tuple<std::uint64_t>{0});
}

}  // namespace aligned_unbound_stores_test
