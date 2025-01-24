/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <fmt/format.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace fixed_array_type_test {

namespace {

using FixedArrayTypeUnit = DefaultFixture;

class FixedArrayTypeTest
  : public FixedArrayTypeUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Type, std::uint32_t, std::string>> {};

class DimTest : public FixedArrayTypeUnit, public ::testing::WithParamInterface<std::uint32_t> {};

class PointTypeTest : public DimTest {};

class PointTypeNegativeDimTest : public DimTest {};

class PointTypeDimMismatchTest : public DimTest {};

class PointTypeMismatchTest
  : public FixedArrayTypeUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Type, std::uint32_t>> {};

INSTANTIATE_TEST_SUITE_P(
  FixedArrayTypeUnit,
  FixedArrayTypeTest,
  ::testing::Values(
    std::make_tuple(legate::uint64(), 20, "uint64[20]") /* element type is a primitive type */,
    std::make_tuple(legate::fixed_array_type(legate::uint16(), 10),
                    10,
                    "uint16[10][10]") /* element type is not a primitive type */,
    std::make_tuple(legate::float64(), 256, "float64[256]") /* // N > 0xFFU */));

INSTANTIATE_TEST_SUITE_P(FixedArrayTypeUnit,
                         PointTypeTest,
                         ::testing::Range(1U, static_cast<std::uint32_t>(LEGATE_MAX_DIM)));

INSTANTIATE_TEST_SUITE_P(FixedArrayTypeUnit,
                         PointTypeNegativeDimTest,
                         ::testing::Values(-1, 0, LEGATE_MAX_DIM + 1));

INSTANTIATE_TEST_SUITE_P(FixedArrayTypeUnit, PointTypeDimMismatchTest, ::testing::Values(-1, 0, 2));

INSTANTIATE_TEST_SUITE_P(FixedArrayTypeUnit,
                         PointTypeMismatchTest,
                         ::testing::Values(std::make_tuple(legate::rect_type(1), 1),
                                           std::make_tuple(legate::string_type(), LEGATE_MAX_DIM)));

void test_fixed_array_type(const legate::Type& type,
                           const legate::Type& element_type,
                           std::uint32_t N,
                           const std::string& to_string)
{
  ASSERT_EQ(type.code(), legate::Type::Code::FIXED_ARRAY);
  ASSERT_EQ(type.size(), element_type.size() * N);
  ASSERT_EQ(type.alignment(), element_type.alignment());
  ASSERT_FALSE(type.variable_size());
  ASSERT_FALSE(type.is_primitive());
  ASSERT_EQ(type.to_string(), to_string);

  // Note: aim to test the copy initialization of Type
  const legate::Type other{type};  // NOLINT(performance-unnecessary-copy-initialization)

  ASSERT_EQ(other, type);

  auto fixed_array_type = type.as_fixed_array_type();

  ASSERT_EQ(fixed_array_type.num_elements(), N);
  ASSERT_EQ(fixed_array_type.element_type(), element_type);
}

}  // namespace

TEST_P(FixedArrayTypeTest, Basic)
{
  const auto [element_type, size, to_string] = GetParam();
  auto fixed_array_type                      = legate::fixed_array_type(element_type, size);

  test_fixed_array_type(fixed_array_type, element_type, size, to_string);
}

TEST_F(FixedArrayTypeUnit, FixedArrayTypeZeroSize)
{
  // N = 0
  ASSERT_NO_THROW(static_cast<void>(legate::fixed_array_type(legate::int64(), 0)));
}

TEST_F(FixedArrayTypeUnit, FixedArrayTypeBadType)
{
  // element type has variable size
  static constexpr auto N = 10;
  ASSERT_THROW(static_cast<void>(legate::fixed_array_type(legate::string_type(), N)),
               std::invalid_argument);
}

TEST_F(FixedArrayTypeUnit, FixedArrayTypeBadCast)
{
  // invalid casts
  ASSERT_THROW(static_cast<void>(legate::uint32().as_fixed_array_type()), std::invalid_argument);
}

TEST_P(PointTypeTest, Basic)
{
  const auto dim = GetParam();
  auto type      = legate::point_type(dim);

  test_fixed_array_type(type, legate::int64(), dim, fmt::format("int64[{}]", dim));
  ASSERT_TRUE(legate::is_point_type(type, dim));
}

TEST_F(FixedArrayTypeUnit, PointTypeSpecific)
{
  // Note: There are several cases in the runtime where 64-bit integers need to be interpreted as 1D
  // points, so we need a more lenient type checking in those cases.
  ASSERT_TRUE(legate::is_point_type(legate::int64(), 1));
}

TEST_P(PointTypeDimMismatchTest, Basic)
{
  ASSERT_FALSE(legate::is_point_type(legate::point_type(1), GetParam()));
}

TEST_P(PointTypeMismatchTest, Basic)
{
  const auto [type, size] = GetParam();

  ASSERT_FALSE(legate::is_point_type(type, size));
}

TEST_P(PointTypeNegativeDimTest, PointType)
{
  ASSERT_THROW(static_cast<void>(legate::point_type(GetParam())), std::out_of_range);
}

}  // namespace fixed_array_type_test
