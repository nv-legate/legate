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

namespace projection_test {

using ProjectionTest = DefaultFixture;

namespace {
constexpr uint64_t BIG_EXTENT   = 100;
constexpr uint64_t SMALL_EXTENT = 2;
}  // namespace

struct ExtraProjectionTester : public legate::LegateTask<ExtraProjectionTester> {
  static constexpr int32_t TASK_ID = 0;
  static void cpu_variant(legate::TaskContext context)
  {
    if (context.is_single_task()) {
      return;
    }
    auto shape1 = context.output(0).shape<3>();
    auto shape2 = context.output(1).shape<3>();
    if (shape1.empty()) {
      return;
    }
    EXPECT_FALSE(shape2.empty());

    auto extents1 = shape1.hi - shape1.lo + legate::Point<3>::ONES();
    auto extents2 = shape2.hi - shape2.lo + legate::Point<3>::ONES();

    // The launch domain is 1D when array partitions have different color shapes
    EXPECT_EQ(context.get_launch_domain().dim, 1);
    // The partitioner favors dimensions of bigger extents over smaller ones
    EXPECT_EQ(extents1[0], SMALL_EXTENT);
    EXPECT_EQ(extents1[2], SMALL_EXTENT);
    EXPECT_EQ(extents2[1], SMALL_EXTENT);
    EXPECT_EQ(extents2[2], SMALL_EXTENT);
  }
};

struct DelinearizeTester : public legate::LegateTask<DelinearizeTester> {
  static constexpr int32_t TASK_ID = 1;
  static void cpu_variant(legate::TaskContext context)
  {
    if (context.is_single_task()) {
      return;
    }
    auto shape1 = context.output(0).shape<3>();
    auto shape2 = context.output(1).shape<3>();
    if (shape1.empty()) {
      return;
    }
    EXPECT_FALSE(shape2.empty());

    auto extents1 = shape1.hi - shape1.lo + legate::Point<3>::ONES();
    auto extents2 = shape2.hi - shape2.lo + legate::Point<3>::ONES();

    // The launch domain is 1D when array partitions have different color shapes
    EXPECT_EQ(context.get_launch_domain().dim, 1);
    // The partitioner favors dimensions of bigger extents over smaller ones
    EXPECT_EQ(extents1[1], SMALL_EXTENT);
    EXPECT_EQ(extents2[2], SMALL_EXTENT);
  }
};

void test_extra_projection(legate::LogicalArray arr1)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_or_create_library("lib_projection_test");
  ExtraProjectionTester::register_variants(library);

  auto arr2 =
    runtime->create_array(legate::Shape{BIG_EXTENT, SMALL_EXTENT, SMALL_EXTENT}, legate::int64());
  auto task = runtime->create_task(library, ExtraProjectionTester::TASK_ID);
  task.add_output(arr1);
  task.add_output(arr2);
  runtime->submit(std::move(task));
}

void test_delinearization(legate::LogicalArray arr1)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_or_create_library("lib_projection_test");
  DelinearizeTester::register_variants(library);

  auto arr2 =
    runtime->create_array(legate::Shape{BIG_EXTENT, BIG_EXTENT, SMALL_EXTENT}, legate::int64());
  auto task = runtime->create_task(library, DelinearizeTester::TASK_ID);
  task.add_output(arr1);
  task.add_output(arr2);
  runtime->submit(std::move(task));
}

TEST_F(ProjectionTest, ExtraProjection1)
{
  test_extra_projection(legate::Runtime::get_runtime()->create_array(
    legate::Shape{SMALL_EXTENT, BIG_EXTENT, SMALL_EXTENT}, legate::int64()));
}

TEST_F(ProjectionTest, ExtraProjection2)
{
  test_extra_projection(legate::Runtime::get_runtime()
                          ->create_array(legate::Shape{SMALL_EXTENT, BIG_EXTENT}, legate::int64())
                          .promote(2, SMALL_EXTENT));
}

TEST_F(ProjectionTest, ExtraProjection3)
{
  test_extra_projection(legate::Runtime::get_runtime()
                          ->create_array(legate::Shape{BIG_EXTENT}, legate::int64())
                          .promote(0, SMALL_EXTENT)
                          .promote(2, SMALL_EXTENT));
}

TEST_F(ProjectionTest, Delinearization1)
{
  test_delinearization(legate::Runtime::get_runtime()->create_array(
    legate::Shape{BIG_EXTENT, SMALL_EXTENT, BIG_EXTENT}, legate::int64()));
}

TEST_F(ProjectionTest, Delinearization2)
{
  test_delinearization(legate::Runtime::get_runtime()
                         ->create_array(legate::Shape{BIG_EXTENT, BIG_EXTENT}, legate::int64())
                         .promote(1, SMALL_EXTENT));
}

}  // namespace projection_test
