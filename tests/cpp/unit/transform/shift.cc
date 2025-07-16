/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace transform_shift_test {

namespace {

using TransformShiftUnit = DefaultFixture;

}  // namespace

TEST_F(TransformShiftUnit, ShiftConvert)
{
  auto transform          = legate::make_internal_shared<legate::detail::Shift>(1, 2);
  auto restrictions_tuple = legate::tuple<legate::detail::Restriction>{
    std::vector<legate::detail::Restriction>({legate::detail::Restriction::ALLOW,
                                              legate::detail::Restriction::FORBID,
                                              legate::detail::Restriction::AVOID})};
  auto restrictions = legate::detail::Restrictions{restrictions_tuple};
  auto result       = transform->convert(restrictions, true /* forbid_fake_dim */);

  ASSERT_THAT(result, ::testing::ContainerEq(restrictions));
  ASSERT_EQ(transform->target_ndim(0), 0);
  ASSERT_TRUE(transform->is_convertible());

  auto dims = legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM>{0};

  ASSERT_NO_THROW(transform->find_imaginary_dims(dims));
  ASSERT_THAT(dims, ::testing::ElementsAre(0));
}

TEST_F(TransformShiftUnit, ShiftConvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::Shift>(1, 2);
  auto expected  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 2, 3};
  auto color     = transform->convert_color(expected);

  ASSERT_EQ(color, expected);
}

TEST_F(TransformShiftUnit, ShiftConvertColorShape)
{
  auto transform = legate::make_internal_shared<legate::detail::Shift>(2, 4);
  auto expected  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 3, 3};
  auto color     = transform->convert_color_shape(expected);

  ASSERT_THAT(color, ::testing::ElementsAre(1, 3, 3));
}

TEST_F(TransformShiftUnit, ShiftConvertExtents)
{
  auto transform = legate::make_internal_shared<legate::detail::Shift>(2, 4);
  auto expected  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 5, 3};
  auto extents   = transform->convert_extents(expected);

  ASSERT_EQ(extents, expected);
}

TEST_F(TransformShiftUnit, ShiftInvertRestrictions)
{
  auto transform = legate::make_internal_shared<legate::detail::Shift>(2, 5);
  auto restrictions_tuple =
    legate::tuple<legate::detail::Restriction>{std::vector<legate::detail::Restriction>(
      {legate::detail::Restriction::ALLOW, legate::detail::Restriction::AVOID})};
  auto restrictions = legate::detail::Restrictions{restrictions_tuple};
  auto result       = transform->invert(restrictions);

  ASSERT_THAT(result, ::testing::ContainerEq(restrictions));
}

TEST_F(TransformShiftUnit, ShiftInvertSymbolicPoint)
{
  auto transform = legate::make_internal_shared<legate::detail::Shift>(2, 5);
  auto point     = legate::proj::create_symbolic_point(LEGATE_MAX_DIM);
  auto result    = transform->invert(point);

  ASSERT_THAT(result, ::testing::ContainerEq(point));
}

TEST_F(TransformShiftUnit, ShiftInvertColor)
{
  auto transform = legate::make_internal_shared<legate::detail::Shift>(3, 3);
  auto expected  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{3, 2, 1};
  auto color     = transform->invert_color(expected);

  ASSERT_EQ(color, expected);
}

TEST_F(TransformShiftUnit, ShiftInvertColorShape)
{
  auto transform = legate::make_internal_shared<legate::detail::Shift>(2, 1);
  auto expected  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1};
  auto color     = transform->invert_color_shape(expected);

  ASSERT_EQ(color, expected);
}

TEST_F(TransformShiftUnit, ShiftInvertExtents)
{
  auto transform = legate::make_internal_shared<legate::detail::Shift>(LEGATE_MAX_DIM, 3);
  auto expected  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{
    legate::detail::tags::size_tag, LEGATE_MAX_DIM, 1};
  auto extents = transform->invert_extents(expected);

  ASSERT_EQ(extents, expected);
}

}  // namespace transform_shift_test
