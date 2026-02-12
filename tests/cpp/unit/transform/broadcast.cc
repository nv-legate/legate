/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/dim_broadcast.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace transform_broadcast_test {

namespace {

using TransformBroadcastUnit = DefaultFixture;
constexpr std::uint64_t EXT  = 42;

}  // namespace

TEST_F(TransformBroadcastUnit, BroadcastTransformDomain)
{
  auto transform = legate::make_internal_shared<legate::detail::DimBroadcast>(1, EXT);
  auto domain    = transform->transform(
    legate::Domain{legate::Rect<3>{legate::Point<3>{1, 2, 3}, legate::Point<3>{5, 7, 8}}});
  auto expected =
    legate::Domain{legate::Rect<3>{legate::Point<3>{1, 0, 3}, legate::Point<3>{5, EXT - 1, 8}}};

  ASSERT_EQ(domain, expected);
}

TEST_F(TransformBroadcastUnit, BroadcastInverseTransform)
{
  auto transform        = legate::make_internal_shared<legate::detail::DimBroadcast>(1, EXT);
  auto affine_transform = transform->inverse_transform(3);
  Legion::DomainAffineTransform expected{};

  // The affine transform should look like this:
  //
  // | 1 0 0 |       | 0 |
  // | 0 0 0 | * p + | 0 |
  // | 0 0 1 |       | 0 |
  expected.transform.m = 3;
  expected.transform.n = 3;
  expected.offset.dim  = 3;
  for (std::int32_t idx = 0; idx < expected.transform.m * expected.transform.n; ++idx) {
    expected.transform.matrix[idx] = 0;
  }
  for (std::int32_t idx = 0; idx < expected.offset.dim; ++idx) {
    expected.offset[idx] = 0;
  }
  expected.transform.matrix[0] = 1;
  expected.transform.matrix[8] = 1;

  ASSERT_EQ(affine_transform, expected);
}

TEST_F(TransformBroadcastUnit, BroadcastConvertRestrictions)
{
  auto transform    = legate::make_internal_shared<legate::detail::DimBroadcast>(1, 3);
  auto restrictions = transform->convert(
    legate::detail::Restrictions{legate::detail::SmallVector{legate::detail::Restriction::ALLOW,
                                                             legate::detail::Restriction::ALLOW,
                                                             legate::detail::Restriction::ALLOW}},
    /*forbid_fake_dim=*/true);
  auto expected =
    legate::detail::Restrictions{legate::detail::SmallVector{legate::detail::Restriction::ALLOW,
                                                             legate::detail::Restriction::FORBID,
                                                             legate::detail::Restriction::ALLOW}};

  ASSERT_EQ(restrictions, expected);
}

TEST_F(TransformBroadcastUnit, BroadcastConvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::DimBroadcast>(1, 3);
  auto color =
    transform->convert_color(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 3});
  auto expected = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 3};

  ASSERT_EQ(color, expected);
  ASSERT_EQ(transform->target_ndim(1), 1);
  ASSERT_TRUE(transform->is_convertible());

  // Also test invert_dims
  auto dims          = legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM>{0, 1};
  auto inverted_dims = transform->invert_dims(std::move(dims));

  ASSERT_THAT(inverted_dims, ::testing::ElementsAre(0, 1));
}

TEST_F(TransformBroadcastUnit, BroadcastConvertColorShape)
{
  auto transform   = legate::make_internal_shared<legate::detail::DimBroadcast>(1, 3);
  auto color_shape = transform->convert_color_shape(
    legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{3, 2, 3});
  auto expected = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{3, 2, 3};

  ASSERT_EQ(color_shape, expected);
}

TEST_F(TransformBroadcastUnit, BroadcastConvertPoint)
{
  auto transform = legate::make_internal_shared<legate::detail::DimBroadcast>(1, 3);
  auto point =
    transform->convert_point(legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{1, 3, 2});
  auto expected = legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{1, 3, 2};

  ASSERT_EQ(point, expected);
}

TEST_F(TransformBroadcastUnit, BroadcastConvertExtents)
{
  auto transform = legate::make_internal_shared<legate::detail::DimBroadcast>(1, 3);
  auto extents =
    transform->convert_extents(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{3, 1, 3});
  auto expected = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{3, 3, 3};

  ASSERT_EQ(extents, expected);
}

TEST_F(TransformBroadcastUnit, BroadcastInvertSymbolicPoint)
{
  auto transform = legate::make_internal_shared<legate::detail::DimBroadcast>(1, 3);
  auto sympoint =
    transform->invert(legate::SymbolicPoint{legate::dimension(0), legate::dimension(1)});
  auto expected = legate::SymbolicPoint{legate::dimension(0), legate::constant(0)};

  ASSERT_EQ(sympoint, expected);
}

TEST_F(TransformBroadcastUnit, BroadcastInvertSymbolicPointOutOfRange)
{
  // dim_ = 1, but point only has 1 element (index 0), so at(1) is out of range
  auto transform = legate::make_internal_shared<legate::detail::DimBroadcast>(1, 3);

  ASSERT_THAT(
    [&] { static_cast<void>(transform->invert(legate::SymbolicPoint{legate::dimension(0)})); },
    ::testing::ThrowsMessage<std::out_of_range>(::testing::HasSubstr("_M_range_check")));
}

TEST_F(TransformBroadcastUnit, BroadcastInvertRestrictions)
{
  auto transform    = legate::make_internal_shared<legate::detail::DimBroadcast>(1, 3);
  auto restrictions = transform->invert(
    legate::detail::Restrictions{legate::detail::SmallVector{legate::detail::Restriction::ALLOW,
                                                             legate::detail::Restriction::ALLOW,
                                                             legate::detail::Restriction::ALLOW}});
  auto expected =
    legate::detail::Restrictions{legate::detail::SmallVector{legate::detail::Restriction::ALLOW,
                                                             legate::detail::Restriction::ALLOW,
                                                             legate::detail::Restriction::ALLOW}};

  ASSERT_EQ(restrictions, expected);
}

TEST_F(TransformBroadcastUnit, BroadcastInvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::DimBroadcast>(1, 3);
  auto color =
    transform->invert_color(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 2});
  auto expected = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 0};

  ASSERT_EQ(color, expected);
}

TEST_F(TransformBroadcastUnit, BroadcastInvertColorNegative)
{
  auto transform = legate::make_internal_shared<legate::detail::DimBroadcast>(1, 3);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    ASSERT_THAT(
      [&] {
        static_cast<void>(
          transform->invert_color(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1}));
      },
      ::testing::ThrowsMessage<std::out_of_range>(::testing::HasSubstr("inplace_vector::at")));
  }
}

TEST_F(TransformBroadcastUnit, BroadcastInvertColorShape)
{
  auto transform   = legate::make_internal_shared<legate::detail::DimBroadcast>(1, 3);
  auto color_shape = transform->invert_color_shape(
    legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{3, 2, 3});
  auto expected = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{3, 1, 3};

  ASSERT_EQ(color_shape, expected);
}

TEST_F(TransformBroadcastUnit, BroadcastInvertPoint)
{
  auto transform = legate::make_internal_shared<legate::detail::DimBroadcast>(1, 3);
  auto point =
    transform->invert_point(legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{1, 3, 2});
  auto expected = legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{1, 0, 2};

  ASSERT_EQ(point, expected);
}

TEST_F(TransformBroadcastUnit, BroadcastInvertExtents)
{
  auto transform = legate::make_internal_shared<legate::detail::DimBroadcast>(1, 3);
  auto extents =
    transform->invert_extents(legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{3, 3, 3});
  auto expected = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{3, 1, 3};

  ASSERT_EQ(extents, expected);
}

TEST_F(TransformBroadcastUnit, BroadcastFindImaginaryDims)
{
  auto transform = legate::make_internal_shared<legate::detail::DimBroadcast>(1, 3);
  auto dims      = legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM>{};
  auto expected  = legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM>{1};

  transform->find_imaginary_dims(dims);
  ASSERT_EQ(dims, expected);
}

}  // namespace transform_broadcast_test
