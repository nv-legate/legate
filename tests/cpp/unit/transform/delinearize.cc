/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform.h>
#include <legate/partitioning/detail/restriction.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace transform_delinearize_test {

namespace {

using TransformDelinearizeUnit = DefaultFixture;

class DelinearizeInvertRestrictions : public TransformDelinearizeUnit,
                                      public ::testing::WithParamInterface<std::tuple<
                                        // Original dimension to delinearize
                                        std::int32_t,
                                        // Sizes of delinearized dimensions
                                        std::vector<std::uint64_t>,
                                        // Input restrictions before the transform
                                        std::vector<legate::detail::Restriction>,
                                        // Expected restrictions after invert
                                        std::vector<legate::detail::Restriction>>> {};

INSTANTIATE_TEST_SUITE_P(
  TransformDelinearizeUnit,
  DelinearizeInvertRestrictions,
  ::testing::Values(
    // Test case 1: Delinearize index 0 to [2, 3], then invert
    std::make_tuple(0,                                   // dim: dimension 0 is delinearized
                    std::vector<std::uint64_t>({2, 3}),  // sizes: delinearize to 2x3 dimensions
                    // Input restrictions: dim_0, dim_0_1, dim_1
                    std::vector<legate::detail::Restriction>({
                      legate::detail::Restriction::ALLOW,   // original dimension 0
                      legate::detail::Restriction::FORBID,  // delinearized dimension 0_1
                      legate::detail::Restriction::AVOID    // original dimension 1
                    }),
                    // Expected after invert: dim_0, dim_1
                    std::vector<legate::detail::Restriction>({
                      legate::detail::Restriction::ALLOW,  // original dimension 0
                      legate::detail::Restriction::AVOID   // original dimension 1
                    })),

    // Test case 2: Delinearize index 1 to [4], then invert
    std::make_tuple(1,                                // dim: dimension 1 is delinearized
                    std::vector<std::uint64_t>({4}),  // sizes: delinearize to 1D of size 4
                    // Input restrictions: dim_0, dim_1, dim_2
                    std::vector<legate::detail::Restriction>({
                      legate::detail::Restriction::ALLOW,   // original dimension 0, also
                                                            // delinearized dimension 0_1
                      legate::detail::Restriction::FORBID,  // original dimension 1
                      legate::detail::Restriction::AVOID    // original dimension 2
                    }),
                    // Expected after invert: dim_0, dim_1, dim_2
                    std::vector<legate::detail::Restriction>({
                      legate::detail::Restriction::ALLOW,   // original dimension 0
                      legate::detail::Restriction::FORBID,  // original dimension 1
                      legate::detail::Restriction::AVOID    // original dimension 2
                    })),

    // Test case 3: Delinearize index 1 to [2, 2], then invert
    std::make_tuple(
      2,                                      // dim: dimension 2 is delinearized
      std::vector<std::uint64_t>({1, 2, 3}),  // sizes: delinearize to 1x2x3 dimensions
      // Input restrictions: dim_0, dim_1, dim_2, dim_2_1, dim_2_2
      std::vector<legate::detail::Restriction>({
        legate::detail::Restriction::AVOID,   // original dimension 0
        legate::detail::Restriction::ALLOW,   // original dimension 1
        legate::detail::Restriction::FORBID,  // original dimension 2
        legate::detail::Restriction::FORBID,  // delinearized dimension 2_1
        legate::detail::Restriction::FORBID   // delinearized dimension 2_2
      }),
      // Expected after invert: dim_0, dim_1, dim_2
      std::vector<legate::detail::Restriction>({
        legate::detail::Restriction::AVOID,  // original dimension 0
        legate::detail::Restriction::ALLOW,  // original dimension 1
        legate::detail::Restriction::FORBID  // original dimension 2
      }))));

}  // namespace

TEST_F(TransformDelinearizeUnit, DelinearizeConvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::Delinearize>(
    2, std::vector<std::uint64_t>({2, 1, 3}));

  ASSERT_THAT(
    [&] { static_cast<void>(transform->convert_color(legate::tuple<std::uint64_t>{2, 1, 6})); },
    ::testing::ThrowsMessage<legate::detail::NonInvertibleTransformation>(
      ::testing::HasSubstr("Non-invertible transformation")));
  ASSERT_FALSE(transform->is_convertible());

  auto dims = std::vector<std::int32_t>({0});

  ASSERT_NO_THROW(transform->find_imaginary_dims(dims));
  ASSERT_EQ(dims, std::vector<std::int32_t>({0}));
}

TEST_F(TransformDelinearizeUnit, DelinearizeConvertColorShape)
{
  auto transform = legate::make_internal_shared<legate::detail::Delinearize>(
    2, std::vector<std::uint64_t>({2, 1, 3}));

  // not convertible for color shape delinearize
  ASSERT_THAT(
    [&] {
      static_cast<void>(transform->convert_color_shape(legate::tuple<std::uint64_t>{2, 1, 6}));
    },
    ::testing::ThrowsMessage<legate::detail::NonInvertibleTransformation>(
      ::testing::HasSubstr("Non-invertible transformation")));
}

TEST_F(TransformDelinearizeUnit, DelinearizeConvertExtents)
{
  auto transform = legate::make_internal_shared<legate::detail::Delinearize>(
    2, std::vector<std::uint64_t>({2, 1, 3}));

  // not convertible for color extents delinearize
  ASSERT_THAT(
    [&] { static_cast<void>(transform->convert_extents(legate::tuple<std::uint64_t>{2, 1, 6})); },
    ::testing::ThrowsMessage<legate::detail::NonInvertibleTransformation>(
      ::testing::HasSubstr("Non-invertible transformation")));
}

TEST_F(TransformDelinearizeUnit, DelinearizeConvertPoint)
{
  auto transform = legate::make_internal_shared<legate::detail::Delinearize>(
    2, std::vector<std::uint64_t>({2, 1, 3}));

  // not convertible for point delinearize
  ASSERT_THAT(
    [&] { static_cast<void>(transform->convert_point(legate::tuple<std::int64_t>{2, -1, 6})); },
    ::testing::ThrowsMessage<legate::detail::NonInvertibleTransformation>(
      ::testing::HasSubstr("Non-invertible transformation")));
}

TEST_P(DelinearizeInvertRestrictions, Basic)
{
  auto [dim, sizes, input_restrictions_vec, expected_restrictions_vec] = GetParam();

  // Create the Delinearize transform
  auto transform = legate::make_internal_shared<legate::detail::Delinearize>(dim, std::move(sizes));

  // Create input restrictions
  auto input_restrictions_tuple =
    legate::tuple<legate::detail::Restriction>{input_restrictions_vec};
  auto input_restrictions = legate::detail::Restrictions{input_restrictions_tuple};

  // Create expected restrictions
  auto expected_restrictions_tuple =
    legate::tuple<legate::detail::Restriction>{expected_restrictions_vec};
  auto expected_restrictions = legate::detail::Restrictions{expected_restrictions_tuple};
  auto result                = transform->invert(input_restrictions);

  ASSERT_THAT(result, ::testing::ContainerEq(expected_restrictions));
}

TEST_F(TransformDelinearizeUnit, DelinearizeInvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::Delinearize>(
    2, std::vector<std::uint64_t>({2, 1, 3}));
  auto result   = transform->invert_color(legate::tuple<std::uint64_t>{2, 1, 7, 0, 0});
  auto expected = legate::tuple<std::uint64_t>{2, 1, 7};

  ASSERT_EQ(result, expected);
}

TEST_F(TransformDelinearizeUnit, DelinearizeInvertColorNegative)
{
  auto transform = legate::make_internal_shared<legate::detail::Delinearize>(
    2, std::vector<std::uint64_t>({2, 1, 3}));

  // dim_ = 2, sum of color[3:4] != 0
  ASSERT_THAT(
    [&] {
      static_cast<void>(transform->invert_color(legate::tuple<std::uint64_t>{2, 2, 1, 3, 6}));
    },
    ::testing::ThrowsMessage<legate::detail::NonInvertibleTransformation>(
      ::testing::HasSubstr("Non-invertible transformation")));
}

TEST_F(TransformDelinearizeUnit, DelinearizeInvertColorShape)
{
  auto transform = legate::make_internal_shared<legate::detail::Delinearize>(
    2, std::vector<std::uint64_t>({2, 1, 3}));
  auto result   = transform->invert_color_shape(legate::tuple<std::uint64_t>{2, 1, 7, 1, 1});
  auto expected = legate::tuple<std::uint64_t>{2, 1, 7};

  ASSERT_EQ(result, expected);
}

TEST_F(TransformDelinearizeUnit, DelinearizeInvertColorShapeNegative)
{
  auto transform = legate::make_internal_shared<legate::detail::Delinearize>(
    2, std::vector<std::uint64_t>({2, 1, 3}));

  // dim_ = 2, volume of color_shape[3:4] != 1
  ASSERT_THAT(
    [&] {
      static_cast<void>(transform->invert_color_shape(legate::tuple<std::uint64_t>{2, 1, 7, 0, 0}));
    },
    ::testing::ThrowsMessage<legate::detail::NonInvertibleTransformation>(
      ::testing::HasSubstr("Non-invertible transformation")));
}

TEST_F(TransformDelinearizeUnit, DelinearizeInvertPoint)
{
  auto transform = legate::make_internal_shared<legate::detail::Delinearize>(
    0, std::vector<std::uint64_t>({2, 1, 3}));
  auto result   = transform->invert_point(legate::tuple<std::int64_t>{2, 2, -2, -1, 6, 7});
  auto expected = legate::tuple<std::int64_t>{6, -1, 6, 7};

  ASSERT_EQ(result, expected);
}

TEST_F(TransformDelinearizeUnit, DelinearizeInvertPointNegative)
{
  auto transform = legate::make_internal_shared<legate::detail::Delinearize>(
    0, std::vector<std::uint64_t>({2, 1, 3}));

  // dim_ = 0, sum of point[1:2] != 0
  ASSERT_THAT(
    [&] {
      static_cast<void>(transform->invert_point(legate::tuple<std::int64_t>{2, 1, 2, -1, 6, 7}));
    },
    ::testing::ThrowsMessage<legate::detail::NonInvertibleTransformation>(
      ::testing::HasSubstr("Non-invertible transformation")));
}

TEST_F(TransformDelinearizeUnit, DelinearizeInvertExtents)
{
  auto transform = legate::make_internal_shared<legate::detail::Delinearize>(
    1, std::vector<std::uint64_t>({2, 1, 3}));
  auto result   = transform->invert_extents(legate::tuple<std::uint64_t>{1, 2, 1, 3, 4});
  auto expected = legate::tuple<std::uint64_t>{1, 6, 4};

  ASSERT_EQ(result, expected);
}

TEST_F(TransformDelinearizeUnit, DelinearizeInvertExtentsNegative)
{
  auto transform = legate::make_internal_shared<legate::detail::Delinearize>(
    0, std::vector<std::uint64_t>({2, 1, 3}));

  // dim_ = 0, extents[dim_ + idx] != sizes_[idx]
  ASSERT_THAT(
    [&] {
      static_cast<void>(transform->invert_extents(legate::tuple<std::uint64_t>{1, 2, 1, 2, 4}));
    },
    ::testing::ThrowsMessage<legate::detail::NonInvertibleTransformation>(
      ::testing::HasSubstr("Non-invertible transformation")));
}

}  // namespace transform_delinearize_test
