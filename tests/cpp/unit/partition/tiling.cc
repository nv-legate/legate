/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partition/tiling.h>

#include <legate.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/transform/promote.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/tuple.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace unit {

namespace {

class TilingTest : public DefaultFixture {
 protected:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    tiling = create_tiling();
  }

 public:
  [[nodiscard]] legate::InternalSharedPtr<legate::detail::Tiling> create_tiling()
  {
    return legate::make_internal_shared<legate::detail::Tiling>(
      legate::Point{4, 4}, legate::Point{2, 2}, legate::Point{1, 1});
  }

  [[nodiscard]] legate::InternalSharedPtr<legate::detail::Tiling> create_overlapped_tiling()
  {
    return legate::make_internal_shared<legate::detail::Tiling>(
      legate::Point{4, 4}, legate::Point{2, 2}, legate::Point{1, 1}, legate::Point{3, 3});
  }

  legate::InternalSharedPtr<legate::detail::Tiling> tiling;
};

}  // namespace

TEST_F(TilingTest, Shape)
{
  auto expected_tile_shape  = legate::Point{4, 4};
  auto expected_color_shape = legate::Point{2, 2};
  auto expected_offsets     = legate::Point{1, 1};
  auto expected_strides     = legate::Point{4, 4};
  EXPECT_EQ(tiling->tile_shape(), expected_tile_shape);
  EXPECT_EQ(tiling->color_shape(), expected_color_shape);
  EXPECT_EQ(tiling->offsets(), expected_offsets);
  EXPECT_EQ(tiling->strides(), expected_strides);
}

TEST_F(TilingTest, Compare)
{
  auto tiling1 = create_tiling();
  ASSERT_EQ(*tiling1, *tiling);

  auto tiling2 = create_overlapped_tiling();
  ASSERT_FALSE(*tiling2 == *tiling);
}

TEST_F(TilingTest, IsConvertible) { ASSERT_TRUE(tiling->is_convertible()); }

TEST_F(TilingTest, LaunchDomain)
{
  ASSERT_TRUE(tiling->has_color_shape());
  auto domain = legate::Rect{legate::NDRect<2>{{0, 0}, {1, 1}}};
  ASSERT_EQ(tiling->launch_domain(), domain);
}

TEST_F(TilingTest, SatisfiesRestrictions)
{
  auto restrictions1 = legate::detail::Restrictions{legate::detail::SmallVector{
    legate::detail::Restriction::ALLOW, legate::detail::Restriction::AVOID}};
  ASSERT_TRUE(restrictions1.are_satisfied_by(*tiling, nullptr));

  auto restrictions2 = legate::detail::Restrictions{legate::detail::SmallVector{
    legate::detail::Restriction::AVOID, legate::detail::Restriction::FORBID}};
  ASSERT_FALSE(restrictions2.are_satisfied_by(*tiling, nullptr));

  auto restrictions3 = legate::detail::Restrictions{legate::detail::SmallVector{
    legate::detail::Restriction::FORBID, legate::detail::Restriction::FORBID}};
  ASSERT_FALSE(restrictions3.are_satisfied_by(*tiling, nullptr));
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
  ASSERT_THROW(static_cast<void>(restrictions1.are_satisfied_by(*tiling, nullptr)),
               std::invalid_argument);

  auto restrictions2 =
    legate::detail::Restrictions{legate::detail::SmallVector{legate::detail::Restriction::ALLOW}};
  ASSERT_THROW(static_cast<void>(restrictions2.are_satisfied_by(*tiling, nullptr)),
               std::invalid_argument);
}

TEST_F(TilingTest, Scale)
{
  auto scale_factor = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 2};
  auto partition    = tiling->scale(scale_factor);
  auto scaled       = dynamic_cast<legate::detail::Tiling*>(partition.get());

  auto expected_tile_shape  = legate::Point{8, 8};
  auto expected_color_shape = legate::Point{2, 2};
  auto expected_offsets     = legate::Point{2, 2};

  ASSERT_EQ(scaled->tile_shape(), expected_tile_shape);
  ASSERT_EQ(scaled->color_shape(), expected_color_shape);
  ASSERT_EQ(scaled->offsets(), expected_offsets);
}

TEST_F(TilingTest, Bloat)
{
  auto low_offsets  = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 1};
  auto high_offsets = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1, 1};
  auto partition    = tiling->bloat(low_offsets, high_offsets);
  auto bloated      = dynamic_cast<legate::detail::Tiling*>(partition.get());

  auto expected_tile_shape  = legate::Point{6, 6};
  auto expected_color_shape = legate::Point{2, 2};
  auto expected_offsets     = legate::Point{0, 0};

  ASSERT_EQ(bloated->tile_shape(), expected_tile_shape);
  ASSERT_EQ(bloated->color_shape(), expected_color_shape);
  ASSERT_EQ(bloated->offsets(), expected_offsets);
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

  auto expected_tile_shape  = transform2->convert_extents(tiling->tile_shape());
  auto expected_color_shape = transform2->convert_color_shape(tiling->color_shape());
  auto expected_offsets     = transform2->convert_point(tiling->offsets());
  auto expected_strides     = transform2->convert_extents(tiling->strides());

  ASSERT_EQ(tiling2->tile_shape(), expected_tile_shape);
  ASSERT_EQ(tiling2->color_shape(), expected_color_shape);
  ASSERT_EQ(tiling2->offsets(), expected_offsets);
  ASSERT_EQ(tiling2->strides(), expected_strides);
}

TEST_F(TilingTest, ToString)
{
  ASSERT_THAT(
    tiling->to_string(),
    ::testing::MatchesRegex(
      R"(Tiling\(tile: \[4, 4\], colors: \[2, 2\], offset: \[1, 1\], strides: \[4, 4\]\))"));
}

TEST_F(TilingTest, Hash)
{
  auto tiling1 = create_tiling();
  auto tiling2 = create_overlapped_tiling();

  ASSERT_EQ(tiling->hash(), tiling1->hash());
  ASSERT_NE(tiling->hash(), tiling2->hash());
}

}  // namespace unit
