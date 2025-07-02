/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/data/detail/logical_store.h>
#include <legate/partitioning/detail/partition.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace unit {

class WeightedTest : public DefaultFixture {
 public:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    weighted = create_weighted();
  }

  [[nodiscard]] legate::InternalSharedPtr<legate::detail::Weighted> create_weighted()
  {
    auto future_map = Legion::FutureMap{};
    auto domain     = Legion::Domain{Legion::Rect<1>{0, 1}};
    return legate::detail::create_weighted(future_map, domain);
  }

  legate::InternalSharedPtr<legate::detail::Weighted> weighted;
};

TEST_F(WeightedTest, Kind)
{
  ASSERT_EQ(weighted->kind(), legate::detail::Partition::Kind::WEIGHTED);
}

TEST_F(WeightedTest, Compare)
{
  auto future_map = Legion::FutureMap{};
  auto domain     = Legion::Domain::NO_DOMAIN;
  auto weighted1  = legate::detail::create_weighted(future_map, domain);

  ASSERT_EQ(*weighted1, *weighted);
  ASSERT_FALSE(*weighted1 < *weighted);
}

TEST_F(WeightedTest, ColorShape)
{
  auto expected_color_shape = legate::tuple<std::uint64_t>{2};
  ASSERT_EQ(weighted->color_shape(), expected_color_shape);
}

TEST_F(WeightedTest, IsCompleteFor)
{
  // This store itself does not actually matter, we just need to create one in order to get a
  // `Storage` reference.
  const auto store =
    legate::Runtime::get_runtime()->create_store(legate::Shape{1}, legate::int32());

  ASSERT_TRUE(weighted->is_complete_for(*store.impl()->get_storage()));
}

TEST_F(WeightedTest, IsConvertible) { ASSERT_FALSE(weighted->is_convertible()); }

TEST_F(WeightedTest, LaunchDomain)
{
  ASSERT_TRUE(weighted->has_launch_domain());

  auto domain = Legion::Domain{Legion::Rect<1>{0, 1}};
  ASSERT_EQ(weighted->launch_domain(), domain);
}

TEST_F(WeightedTest, IsDisjointFor)
{
  auto domain1 = Legion::Domain::NO_DOMAIN;
  ASSERT_TRUE(weighted->is_disjoint_for(domain1));

  auto domain2 = Legion::Domain{Legion::Rect<1>{0, 1}};
  ASSERT_TRUE(weighted->is_disjoint_for(domain2));

  auto domain3 = Legion::Domain{Legion::Rect<1>{0, 2}};
  ASSERT_FALSE(weighted->is_disjoint_for(domain3));
}

TEST_F(WeightedTest, SatisfiesRestrictions)
{
  auto restrictions1 = legate::detail::Restrictions{legate::detail::Restriction::ALLOW};
  ASSERT_TRUE(weighted->satisfies_restrictions(restrictions1));

  auto restrictions2 = legate::detail::Restrictions{legate::detail::Restriction::AVOID};
  ASSERT_TRUE(weighted->satisfies_restrictions(restrictions2));

  auto restrictions3 = legate::detail::Restrictions{legate::detail::Restriction::FORBID};
  ASSERT_FALSE(weighted->satisfies_restrictions(restrictions3));
}

TEST_F(WeightedTest, SatisfiesRestrictionsNegative)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Sizes are only checked in debug builds";
  }

  auto restrictions1 = legate::detail::Restrictions{legate::detail::Restriction::ALLOW,
                                                    legate::detail::Restriction::AVOID};
  ASSERT_THROW(static_cast<void>(weighted->satisfies_restrictions(restrictions1)),
               std::invalid_argument);

  auto restrictions2 = legate::detail::Restrictions{};
  ASSERT_THROW(static_cast<void>(weighted->satisfies_restrictions(restrictions2)),
               std::invalid_argument);
}

TEST_F(WeightedTest, Scale)
{
  // Scale is not implemented for weighted partitions
  auto factors = legate::tuple<std::uint64_t>{2, 2};
  ASSERT_THROW(static_cast<void>(weighted->scale(factors)), std::runtime_error);
}

TEST_F(WeightedTest, Bloat)
{
  // Bloat is not implemented for weighted partitions
  auto low_offsets  = legate::tuple<std::uint64_t>{0, 0};
  auto high_offsets = legate::tuple<std::uint64_t>{1, 1};
  ASSERT_THROW(static_cast<void>(weighted->bloat(low_offsets, high_offsets)), std::runtime_error);
}

TEST_F(WeightedTest, Convert)
{
  // Test identity transformation
  auto transform1    = legate::make_internal_shared<legate::detail::TransformStack>();
  auto weighted1     = weighted->convert(weighted, transform1);
  auto weighted1_raw = weighted1.get();
  auto weighted_raw  = weighted.get();
  ASSERT_EQ(weighted1_raw, weighted_raw);

  // Test non-identity transformation
  auto dim        = 2;
  auto sizes      = std::vector<std::uint64_t>{2, 1};
  auto transform2 = legate::make_internal_shared<legate::detail::TransformStack>(
    std::make_unique<legate::detail::Delinearize>(dim, std::move(sizes)), std::move(transform1));
  ASSERT_THROW(static_cast<void>(weighted->convert(weighted, transform2)),
               legate::detail::NonInvertibleTransformation);
}

TEST_F(WeightedTest, Invert)
{
  // Test identity transformation
  auto transform1    = legate::make_internal_shared<legate::detail::TransformStack>();
  auto weighted1     = weighted->invert(weighted, transform1);
  auto weighted1_raw = weighted1.get();
  auto weighted_raw  = weighted.get();
  ASSERT_EQ(weighted1_raw, weighted_raw);

  // Test non-identity transformation
  auto dim        = 1;
  auto sizes      = std::vector<std::uint64_t>{1};
  auto transform2 = legate::make_internal_shared<legate::detail::TransformStack>(
    std::make_unique<legate::detail::Delinearize>(dim, std::move(sizes)), std::move(transform1));

  auto weighted2            = weighted->invert(weighted, transform2);
  auto expected_color_shape = transform2->invert_color_shape(weighted2->color_shape());
  ASSERT_EQ(weighted2->color_shape(), expected_color_shape);
}

TEST_F(WeightedTest, ToString)
{
  constexpr std::string_view weighted_str = "Weighted({})";

  std::stringstream ss;
  ss << *weighted;

  ASSERT_EQ(ss.str(), weighted_str);
  ASSERT_EQ(weighted->to_string(), weighted_str);
}

}  // namespace unit
