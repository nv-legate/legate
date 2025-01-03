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

namespace logical_store_transform {

namespace {

constexpr std::uint64_t EXT = 17;
constexpr std::int64_t VAL  = 42;

}  // namespace

class InitTask : public legate::LegateTask<InitTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context)
  {
    auto output = context.output(0);
    auto buffer = output.data().create_output_buffer<std::int64_t, 1>(legate::Point<1>{EXT}, true);
    for (std::int64_t idx = 0; idx < static_cast<std::int64_t>(EXT); ++idx) {
      buffer[idx] = VAL;
    }
  }
};

class CopyTask : public legate::LegateTask<CopyTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{1};

  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0);
    auto output = context.output(0);
    auto value  = context.scalar(0).value<std::int64_t>();

    auto shape = output.shape<1>();

    if (shape.empty()) {
      return;
    }

    auto in_acc  = input.data().read_accessor<std::int64_t, 1>();
    auto out_acc = output.data().write_accessor<std::int64_t, 1>();

    for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
      out_acc[*it] = in_acc[*it] + value;
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_transform_on_weighted_partition";
  static void registration_callback(legate::Library library)
  {
    InitTask::register_variants(library);
    CopyTask::register_variants(library);
  }
};

using LogicalStoreTransform = RegisterOnceFixture<Config>;

// NOLINTBEGIN(readability-magic-numbers)

TEST_F(LogicalStoreTransform, SliceBug1)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{4, 3}, legate::int64());

  auto test_slice = [&store](auto&& slice) {
    EXPECT_EQ(slice.volume(), 0);
    EXPECT_FALSE(slice.transformed());
    EXPECT_EQ(slice.type(), store.type());
    EXPECT_FALSE(slice.overlaps(store));
    EXPECT_EQ(slice.dim(), store.dim());
  };
  test_slice(store.slice(1, legate::Slice{-9, -8}));
  test_slice(store.slice(1, legate::Slice{-8, -10}));
  test_slice(store.slice(1, legate::Slice{1, 1}));
}

TEST_F(LogicalStoreTransform, SliceBug2)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{4, 3}, legate::int64());

  auto test_slice = [&store](auto&& slice) {
    EXPECT_EQ(slice.volume(), 0);
    EXPECT_FALSE(slice.transformed());
    EXPECT_EQ(slice.type(), store.type());
    EXPECT_FALSE(slice.overlaps(store));
    EXPECT_EQ(slice.dim(), store.dim());
  };

  test_slice(store.slice(1, legate::Slice{-1, 0}));
  test_slice(store.slice(1, legate::Slice{-1, 1}));
  test_slice(store.slice(1, legate::Slice{10, 8}));
}

namespace {

void init(const legate::LogicalArray& output)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  // Dummy argument to get the task parallelized
  auto dummy = runtime->create_array(legate::Shape{EXT}, legate::int64());

  auto task1 = runtime->create_task(library, InitTask::TASK_ID);
  task1.add_output(output);
  task1.add_output(dummy);
  runtime->submit(std::move(task1));
}

void add_and_copy(const legate::LogicalArray& output,
                  const legate::LogicalArray& input,
                  std::int64_t index)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto task  = runtime->create_task(library, CopyTask::TASK_ID);
  auto part1 = task.add_input(input);
  auto part2 = task.add_output(output.project(1, index));
  task.add_scalar_arg(index);
  task.add_constraint(legate::align(part1, part2));
  runtime->submit(std::move(task));
}

}  // namespace

TEST_F(LogicalStoreTransform, WeightedBug1)
{
  auto runtime = legate::Runtime::get_runtime();

  auto arr1 = runtime->create_array(legate::int64());
  init(arr1);

  auto arr2 = runtime->create_array(legate::Shape{arr1.shape()[0], 3}, legate::int64());
  for (std::int64_t idx = 0; idx < 3; ++idx) {
    add_and_copy(arr2, arr1, idx);
  }

  auto store = arr2.data().get_physical_store();
  auto acc   = store.read_accessor<std::int64_t, 2>();
  auto shape = store.shape<2>();

  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    EXPECT_EQ(acc[*it] - (*it)[1], VAL);
  }
}

// NOLINTEND(readability-magic-numbers)

}  // namespace logical_store_transform
