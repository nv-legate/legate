/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partition/tiling.h>

#include <legate.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/transform/delinearize.h>
#include <legate/data/detail/transform/promote.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/utilities/detail/small_vector.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace unit {

class TilingTest : public DefaultFixture {
 public:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    tiling = create_tiling();
  }

  [[nodiscard]] legate::InternalSharedPtr<legate::detail::Tiling> create_tiling()
  {
    auto tile_shape  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{4, 4};
    auto color_shape = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 2};
    auto offsets     = legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{1, 1};
    return legate::detail::create_tiling(tile_shape, color_shape, offsets);
  }

  [[nodiscard]] legate::InternalSharedPtr<legate::detail::Tiling> create_overlapped_tiling()
  {
    auto tile_shape  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{4, 4};
    auto color_shape = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 2};
    auto offsets     = legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{1, 1};
    auto strides     = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{3, 3};
    return legate::detail::create_tiling(tile_shape, color_shape, offsets, strides);
  }

  legate::InternalSharedPtr<legate::detail::Tiling> tiling;
};

TEST_F(TilingTest, Shape)
{
  auto expected_tile_shape  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{4, 4};
  auto expected_color_shape = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 2};
  auto expected_offsets     = legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{1, 1};
  auto expected_strides     = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{4, 4};
  EXPECT_EQ((legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling->tile_shape()}),
            expected_tile_shape);
  EXPECT_EQ((legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling->color_shape()}),
            expected_color_shape);
  EXPECT_EQ((legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{tiling->offsets()}),
            expected_offsets);
  EXPECT_EQ((legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling->strides()}),
            expected_strides);
}

TEST_F(TilingTest, Compare)
{
  auto tiling1 = create_tiling();
  ASSERT_EQ(*tiling1, *tiling);

  auto tiling2 = create_overlapped_tiling();
  ASSERT_FALSE(*tiling2 == *tiling);
}

TEST_F(TilingTest, IsCompleteFor)
{
  auto runtime = legate::Runtime::get_runtime();

  constexpr auto dim1 = 8;
  auto shape1         = legate::Shape{dim1, dim1};
  auto store1         = runtime->create_store(shape1, legate::int32());
  ASSERT_TRUE(tiling->is_complete_for(*store1.impl()->get_storage()));

  constexpr auto dim2 = 10;
  auto shape2         = legate::Shape{dim2, dim2};
  auto store2         = runtime->create_store(shape2, legate::int32());
  ASSERT_FALSE(tiling->is_complete_for(*store2.impl()->get_storage()));
}

TEST_F(TilingTest, IsConvertible) { ASSERT_TRUE(tiling->is_convertible()); }

TEST_F(TilingTest, LaunchDomain)
{
  ASSERT_TRUE(tiling->has_launch_domain());
  auto domain = Legion::Domain{Legion::Rect<2>{{0, 0}, {1, 1}}};
  ASSERT_EQ(tiling->launch_domain(), domain);
}

TEST_F(TilingTest, SatisfiesRestrictions)
{
  auto restrictions1 = legate::detail::Restrictions{legate::detail::SmallVector{
    legate::detail::Restriction::ALLOW, legate::detail::Restriction::AVOID}};
  ASSERT_TRUE(restrictions1.are_satisfied_by(*tiling));

  auto restrictions2 = legate::detail::Restrictions{legate::detail::SmallVector{
    legate::detail::Restriction::AVOID, legate::detail::Restriction::FORBID}};
  ASSERT_FALSE(restrictions2.are_satisfied_by(*tiling));

  auto restrictions3 = legate::detail::Restrictions{legate::detail::SmallVector{
    legate::detail::Restriction::FORBID, legate::detail::Restriction::FORBID}};
  ASSERT_FALSE(restrictions3.are_satisfied_by(*tiling));
}

TEST_F(TilingTest, SatisfiesRestrictionsNegative)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Sizes are only checked in debug builds";
  }

  auto restrictions1 =
    legate::detail::Restrictions{legate::detail::SmallVector{legate::detail::Restriction::ALLOW,
                                                             legate::detail::Restriction::AVOID,
                                                             legate::detail::Restriction::FORBID}};
  ASSERT_THROW(static_cast<void>(restrictions1.are_satisfied_by(*tiling)), std::invalid_argument);

  auto restrictions2 =
    legate::detail::Restrictions{legate::detail::SmallVector{legate::detail::Restriction::ALLOW}};
  ASSERT_THROW(static_cast<void>(restrictions2.are_satisfied_by(*tiling)), std::invalid_argument);
}

TEST_F(TilingTest, IsDisjointFor)
{
  // Test none-overlapping case
  auto domain1 = Legion::Domain::NO_DOMAIN;
  ASSERT_TRUE(tiling->is_disjoint_for(domain1));

  auto domain2 = Legion::Domain{Legion::Rect<2>{{0, 0}, {1, 1}}};
  ASSERT_TRUE(tiling->is_disjoint_for(domain2));

  auto domain3 = Legion::Domain{Legion::Rect<2>{{0, 0}, {2, 2}}};
  ASSERT_FALSE(tiling->is_disjoint_for(domain3));

  // Test overlapping case
  auto tiling_overlapped = create_overlapped_tiling();
  ASSERT_FALSE(tiling_overlapped->is_disjoint_for(domain1));
}

TEST_F(TilingTest, Scale)
{
  auto scale_factor = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 2};
  auto partition    = tiling->scale(scale_factor);
  auto scaled       = dynamic_cast<legate::detail::Tiling*>(partition.get());

  auto expected_tile_shape  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{8, 8};
  auto expected_color_shape = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 2};
  auto expected_offsets     = legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{2, 2};

  ASSERT_EQ((legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{scaled->tile_shape()}),
            expected_tile_shape);
  ASSERT_EQ((legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{scaled->color_shape()}),
            expected_color_shape);
  ASSERT_EQ((legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{scaled->offsets()}),
            expected_offsets);
}

TEST_F(TilingTest, Bloat)
{
  auto low_offsets  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 1};
  auto high_offsets = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 1};
  auto partition    = tiling->bloat(low_offsets, high_offsets);
  auto bloated      = dynamic_cast<legate::detail::Tiling*>(partition.get());

  auto expected_tile_shape  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{6, 6};
  auto expected_color_shape = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 2};
  auto expected_offsets     = legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{0, 0};

  ASSERT_EQ((legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{bloated->tile_shape()}),
            expected_tile_shape);
  ASSERT_EQ((legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{bloated->color_shape()}),
            expected_color_shape);
  ASSERT_EQ((legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{bloated->offsets()}),
            expected_offsets);
}

TEST_F(TilingTest, Convert)
{
  // Test identity transformation
  auto transform1  = legate::make_internal_shared<legate::detail::TransformStack>();
  auto tiling1     = tiling->convert(tiling, transform1);
  auto tiling1_raw = tiling1.get();
  auto tiling_raw  = tiling.get();
  ASSERT_EQ(tiling1_raw, tiling_raw);

  // Test non-identity transformation
  auto extra_dim  = 2;
  auto dim_size   = 4;
  auto transform2 = legate::make_internal_shared<legate::detail::TransformStack>(
    std::make_unique<legate::detail::Promote>(extra_dim, dim_size), std::move(transform1));
  auto partition2 = tiling->convert(tiling, transform2);
  auto tiling2    = dynamic_cast<legate::detail::Tiling*>(partition2.get());

  auto expected_tile_shape = transform2->convert_extents(
    legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling->tile_shape()});
  auto expected_color_shape = transform2->convert_color_shape(
    legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling->color_shape()});
  auto expected_offsets = transform2->convert_point(
    legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{tiling->offsets()});
  auto expected_strides = transform2->convert_extents(
    legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling->strides()});

  ASSERT_EQ((legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling2->tile_shape()}),
            expected_tile_shape);
  ASSERT_EQ((legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling2->color_shape()}),
            expected_color_shape);
  ASSERT_EQ((legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{tiling2->offsets()}),
            expected_offsets);
  ASSERT_EQ((legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling2->strides()}),
            expected_strides);
}

TEST_F(TilingTest, Invert)
{
  // Test identity transformation
  auto transform1  = legate::make_internal_shared<legate::detail::TransformStack>();
  auto tiling1     = tiling->invert(tiling, transform1);
  auto tiling1_raw = tiling1.get();
  auto tiling_raw  = tiling.get();
  ASSERT_EQ(tiling1_raw, tiling_raw);

  // Test non-identity transformation
  auto dim        = 1;
  auto sizes      = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1};
  auto transform2 = legate::make_internal_shared<legate::detail::TransformStack>(
    std::make_unique<legate::detail::Delinearize>(dim, std::move(sizes)), std::move(transform1));

  auto expected_tile_shape = transform2->invert_extents(
    legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling->tile_shape()});
  auto expected_color_shape = transform2->invert_color_shape(
    legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling->color_shape()});
  auto expected_offsets = transform2->invert_point(
    legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{tiling->offsets()});
  auto expected_strides = transform2->invert_extents(
    legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling->strides()});

  auto partition2 = tiling->invert(tiling, transform2);
  auto tiling2    = dynamic_cast<legate::detail::Tiling*>(partition2.get());

  ASSERT_EQ((legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling2->tile_shape()}),
            expected_tile_shape);
  ASSERT_EQ((legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling2->color_shape()}),
            expected_color_shape);
  ASSERT_EQ((legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{tiling2->offsets()}),
            expected_offsets);
  ASSERT_EQ((legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{tiling2->strides()}),
            expected_strides);
}

TEST_F(TilingTest, ToString)
{
  ASSERT_THAT(
    tiling->to_string(),
    ::testing::MatchesRegex(
      R"(Tiling\(tile: \[4, 4\], colors: \[2, 2\], offset: \[1, 1\], strides: \[4, 4\]\))"));
}

TEST_F(TilingTest, ChildExtents)
{
  auto extents = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 2};

  // Input extents are within the selected tile
  auto color1            = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{0, 0};
  auto child_extents1    = tiling->get_child_extents(extents, color1);
  auto expected_extents1 = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 1};
  ASSERT_EQ(child_extents1, expected_extents1);

  // Input extents are out of the selected tile
  auto color2            = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 1};
  auto child_extents2    = tiling->get_child_extents(extents, color2);
  auto expected_extents2 = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{0, 0};
  ASSERT_EQ(child_extents2, expected_extents2);
}

TEST_F(TilingTest, ChildExtentsNegative)
{
  auto extents = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 2};
  auto color   = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 2};
  ASSERT_THROW(static_cast<void>(tiling->get_child_extents(extents, color)), std::invalid_argument);
}

TEST_F(TilingTest, ChildOffsets)
{
  auto color            = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 1};
  auto child_offsets    = tiling->get_child_offsets(color);
  auto expected_offsets = legate::detail::SmallVector<std::int64_t, LEGATE_MAX_DIM>{5, 5};
  ASSERT_EQ(child_offsets, expected_offsets);
}

TEST_F(TilingTest, ChildOffsetsNegative)
{
  auto color = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 2};
  ASSERT_THROW(static_cast<void>(tiling->get_child_offsets(color)), std::invalid_argument);
}

TEST_F(TilingTest, HasColor)
{
  auto color1     = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{0, 0};
  auto has_color1 = tiling->has_color(color1);
  ASSERT_TRUE(has_color1);

  auto color2     = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 1};
  auto has_color2 = tiling->has_color(color2);
  ASSERT_TRUE(has_color2);

  auto color3     = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 2};
  auto has_color3 = tiling->has_color(color3);
  ASSERT_FALSE(has_color3);
}

TEST_F(TilingTest, Hash)
{
  auto tiling1 = create_tiling();
  auto tiling2 = create_overlapped_tiling();

  ASSERT_EQ(tiling->hash(), tiling1->hash());
  ASSERT_NE(tiling->hash(), tiling2->hash());
}

}  // namespace unit
