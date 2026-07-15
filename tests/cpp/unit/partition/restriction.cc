/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/restriction.h>

#include <legate.h>

#include <legate/partitioning/detail/partition/tiling.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace unit {

namespace {

using Restriction  = legate::detail::Restriction;
using Restrictions = legate::detail::Restrictions;

[[nodiscard]] legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents(
  std::initializer_list<std::uint64_t> values)
{
  return legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{values};
}

[[nodiscard]] legate::Span<const std::uint64_t> span(
  const legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>& values)
{
  return {values.data(), values.size()};
}

[[nodiscard]] Restrictions make_restrictions(std::initializer_list<Restriction> restrictions,
                                             bool require_invertible = false)
{
  auto result = Restrictions{legate::detail::SmallVector<Restriction>{restrictions}};
  result.set_require_invertible(require_invertible);
  return result;
}

[[nodiscard]] legate::InternalSharedPtr<legate::detail::Tiling> create_tiling(
  std::initializer_list<std::uint64_t> tile_extents,
  std::initializer_list<std::uint64_t> color_extents)
{
  auto tile_shape  = extents(tile_extents);
  auto color_shape = extents(color_extents);
  return legate::detail::create_tiling(tile_shape, color_shape);
}

}  // namespace

TEST(RestrictionsUnit, EqualityIncludesInvertibility)
{
  auto lhs = make_restrictions({Restriction::ALLOW});
  auto rhs = make_restrictions({Restriction::ALLOW}, /*require_invertible=*/true);

  ASSERT_FALSE(lhs == rhs);
}

TEST(RestrictionsUnit, ApplyMinimumExtentsKeepsPerDimensionMaximum)
{
  auto restrictions =
    Restrictions{legate::detail::SmallVector<Restriction>{Restriction::ALLOW, Restriction::ALLOW},
                 extents({4, 1})};
  const auto new_minimum_extents = extents({2, 3});
  const auto satisfied_shape     = extents({8, 9});
  const auto too_small_dim0      = extents({6, 9});
  const auto too_small_dim1      = extents({8, 6});
  const auto color_shape         = extents({2, 3});

  restrictions.apply_minimum_extents(span(new_minimum_extents));

  ASSERT_TRUE(restrictions.minimum_extents_satisfied_by(span(satisfied_shape), span(color_shape)));
  ASSERT_FALSE(restrictions.minimum_extents_satisfied_by(span(too_small_dim0), span(color_shape)));
  ASSERT_FALSE(restrictions.minimum_extents_satisfied_by(span(too_small_dim1), span(color_shape)));
}

TEST(RestrictionsUnit, AreSatisfiedByAllowsForbiddenSingletonColorExtent)
{
  const auto tiling = create_tiling({4, 4}, {1, 2});
  auto restrictions = make_restrictions({Restriction::FORBID, Restriction::ALLOW});

  ASSERT_TRUE(restrictions.are_satisfied_by(*tiling, nullptr));
}

TEST(RestrictionsUnit, JoinReturnsCombinedRestrictions)
{
  auto lhs      = make_restrictions({Restriction::ALLOW, Restriction::FORBID});
  auto rhs      = make_restrictions({Restriction::AVOID, Restriction::ALLOW},
                                    /*require_invertible=*/true);
  auto expected = make_restrictions({Restriction::AVOID, Restriction::FORBID},
                                    /*require_invertible=*/true);

  ASSERT_EQ(lhs.join(rhs), expected);
  ASSERT_FALSE(lhs == expected);
}

TEST(RestrictionsUnit, JoinInplaceRejectsDifferentSizes)
{
  auto lhs = make_restrictions({Restriction::ALLOW});
  auto rhs = make_restrictions({Restriction::ALLOW, Restriction::AVOID});

  ASSERT_THAT([&] { lhs.join_inplace(rhs); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Restrictions must have the same size")));
}

TEST(RestrictionsUnit, ToProjectionEmbedsForbiddenDimensions)
{
  auto restrictions =
    make_restrictions({Restriction::ALLOW, Restriction::FORBID, Restriction::AVOID});
  auto expected =
    legate::SymbolicPoint{legate::dimension(0), legate::constant(0), legate::dimension(1)};

  ASSERT_EQ(restrictions.to_projection(), expected);
}

}  // namespace unit
