/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace logical_store_project_test {

namespace {

using LogicalStoreProjectUnit = DefaultFixture;

constexpr std::int32_t SCALAR_VALUE = 10;

class ProjectBoundStoreTest
  : public LogicalStoreProjectUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Shape, std::int32_t, std::int64_t, std::vector<std::uint64_t>>> {};

class ProjectScalarStoreTest
  : public LogicalStoreProjectUnit,
    public ::testing::WithParamInterface<
      std::tuple<std::int32_t, std::int64_t, std::vector<std::uint64_t>>> {};

class NegativeProjectStoreDimTest : public LogicalStoreProjectUnit,
                                    public ::testing::WithParamInterface<std::int32_t> {};

class NegativeProjectBoundStoreIndexTest
  : public LogicalStoreProjectUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Shape, std::int32_t, std::int64_t>> {};

class NegativeProjectScalarStoreIndexTest : public LogicalStoreProjectUnit,
                                            public ::testing::WithParamInterface<std::int64_t> {};

INSTANTIATE_TEST_SUITE_P(
  LogicalStoreProjectUnit,
  ProjectBoundStoreTest,
  ::testing::Values(std::make_tuple(legate::Shape{4, 3}, 0, 1L, std::vector<std::uint64_t>({3})),
                    std::make_tuple(legate::Shape{4, 3}, 0, 3L, std::vector<std::uint64_t>({3})),
                    std::make_tuple(legate::Shape{4, 3}, 1, 0L, std::vector<std::uint64_t>({4})),
                    std::make_tuple(legate::Shape{4, 3}, 1, 2L, std::vector<std::uint64_t>({4})),
                    std::make_tuple(legate::Shape{0, 4}, 1, 0L, std::vector<std::uint64_t>({0})),
                    std::make_tuple(legate::Shape{4}, 0, 1L, std::vector<std::uint64_t>({}))));

INSTANTIATE_TEST_SUITE_P(LogicalStoreProjectUnit,
                         ProjectScalarStoreTest,
                         ::testing::Values(std::make_tuple(0, 0L, std::vector<std::uint64_t>({}))));

INSTANTIATE_TEST_SUITE_P(LogicalStoreProjectUnit,
                         NegativeProjectStoreDimTest,
                         ::testing::Values(-1, LEGATE_MAX_DIM));

INSTANTIATE_TEST_SUITE_P(LogicalStoreProjectUnit,
                         NegativeProjectBoundStoreIndexTest,
                         ::testing::Values(std::make_tuple(legate::Shape{4, 3}, 0, 4L),
                                           std::make_tuple(legate::Shape{4, 3}, 0, -1L),
                                           std::make_tuple(legate::Shape{4, 3}, 1, 3L),
                                           std::make_tuple(legate::Shape{4, 3}, 1, -3L),
                                           std::make_tuple(legate::Shape{0}, 0, 1L),
                                           std::make_tuple(legate::Shape{0}, 0, -1L),
                                           std::make_tuple(legate::Shape{}, 0, 1L),
                                           std::make_tuple(legate::Shape{}, 0, 0L),
                                           std::make_tuple(legate::Shape{}, 0, -1L)));

INSTANTIATE_TEST_SUITE_P(LogicalStoreProjectUnit,
                         NegativeProjectScalarStoreIndexTest,
                         ::testing::Values(1L, -1L));

}  // namespace

TEST_P(ProjectBoundStoreTest, Basic)
{
  const auto [shape, dim, index, project_shape] = GetParam();
  auto runtime                                  = legate::Runtime::get_runtime();
  auto store                                    = runtime->create_store(shape, legate::int64());
  auto project                                  = store.project(dim, index);

  ASSERT_EQ(project.extents().data(), project_shape);
  ASSERT_TRUE(project.transformed());
  ASSERT_EQ(project.type(), store.type());
  ASSERT_TRUE(project.overlaps(store));
  ASSERT_EQ(project.dim(), store.dim() - 1);
}

TEST_P(ProjectScalarStoreTest, ProjectScalarStore)
{
  const auto [dim, index, project_shape] = GetParam();
  auto runtime                           = legate::Runtime::get_runtime();
  auto store                             = runtime->create_store(legate::Scalar{SCALAR_VALUE});
  auto project                           = store.project(dim, index);

  ASSERT_EQ(project.extents().data(), project_shape);
  ASSERT_TRUE(project.transformed());
  ASSERT_EQ(project.type(), store.type());
  ASSERT_TRUE(project.overlaps(store));
  ASSERT_EQ(project.dim(), 0);
}

TEST_P(NegativeProjectStoreDimTest, BoundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{4}, legate::int64());

  ASSERT_THROW(static_cast<void>(store.project(GetParam(), 1)), std::invalid_argument);
}

TEST_P(NegativeProjectBoundStoreIndexTest, BoundStore)
{
  const auto [shape, dim, index] = GetParam();
  auto runtime                   = legate::Runtime::get_runtime();
  auto store                     = runtime->create_store(shape, legate::int64());

  ASSERT_THROW(static_cast<void>(store.project(dim, index)), std::invalid_argument);
}

TEST_P(NegativeProjectStoreDimTest, ScalarStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{SCALAR_VALUE});

  ASSERT_THROW(static_cast<void>(store.project(GetParam(), 0)), std::invalid_argument);
}

TEST_P(NegativeProjectScalarStoreIndexTest, ScalarStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Scalar{SCALAR_VALUE});

  ASSERT_THROW(static_cast<void>(store.project(0, GetParam())), std::invalid_argument);
}

TEST_F(LogicalStoreProjectUnit, UnboundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::int64(), LEGATE_MAX_DIM);

  ASSERT_THROW(static_cast<void>(store.project(0, 0)), std::invalid_argument);
}

}  // namespace logical_store_project_test
