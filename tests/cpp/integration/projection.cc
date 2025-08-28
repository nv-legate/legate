/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace projection_test {

namespace {

constexpr std::uint64_t BIGGER_EXTENT = 200;
constexpr std::uint64_t BIG_EXTENT    = 100;
constexpr std::uint64_t SMALL_EXTENT  = 2;

struct ExtraProjectionTester : public legate::LegateTask<ExtraProjectionTester> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

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
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

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

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "lib_projection_test";

  static void registration_callback(legate::Library library)
  {
    ExtraProjectionTester::register_variants(library);
    DelinearizeTester::register_variants(library);
  }
};

class ProjectionTest : public RegisterOnceFixture<Config> {};

void test_extra_projection(const legate::LogicalArray& arr1)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto arr2 =
    runtime->create_array(legate::Shape{BIG_EXTENT, SMALL_EXTENT, SMALL_EXTENT}, legate::int64());
  auto task = runtime->create_task(library, ExtraProjectionTester::TASK_CONFIG.task_id());
  task.add_output(arr1);
  task.add_output(arr2);
  runtime->submit(std::move(task));
}

void test_delinearization(const legate::LogicalArray& arr1)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto arr2 =
    runtime->create_array(legate::Shape{BIG_EXTENT, BIG_EXTENT, SMALL_EXTENT}, legate::int64());
  auto task = runtime->create_task(library, DelinearizeTester::TASK_CONFIG.task_id());
  task.add_output(arr1);
  task.add_output(arr2);
  runtime->submit(std::move(task));
}

}  // namespace

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
    legate::Shape{BIG_EXTENT, SMALL_EXTENT, BIGGER_EXTENT}, legate::int64()));
}

TEST_F(ProjectionTest, Delinearization2)
{
  test_delinearization(legate::Runtime::get_runtime()
                         ->create_array(legate::Shape{BIG_EXTENT, BIGGER_EXTENT}, legate::int64())
                         .promote(1, SMALL_EXTENT));
}

}  // namespace projection_test
