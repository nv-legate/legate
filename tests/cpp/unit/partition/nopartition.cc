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

class NoPartitionTest : public DefaultFixture {
 public:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    nopartition = legate::detail::create_no_partition();
  }

  legate::InternalSharedPtr<legate::detail::NoPartition> nopartition;
};

TEST_F(NoPartitionTest, Kind)
{
  ASSERT_EQ(nopartition->kind(), legate::detail::Partition::Kind::NO_PARTITION);
}

TEST_F(NoPartitionTest, IsCompleteFor)
{
  // This store itself does not actually matter, we just need to create one in order to get a
  // `Storage` reference.
  const auto store =
    legate::Runtime::get_runtime()->create_store(legate::Shape{1}, legate::int32());

  ASSERT_TRUE(nopartition->is_complete_for(*store.impl()->get_storage()));
}

TEST_F(NoPartitionTest, IsConvertible) { ASSERT_TRUE(nopartition->is_convertible()); }

TEST_F(NoPartitionTest, SatisfiesRestrictions)
{
  auto restrictions1 = legate::detail::Restrictions{};
  ASSERT_TRUE(nopartition->satisfies_restrictions(restrictions1));

  auto restrictions2 = legate::detail::Restrictions{legate::detail::Restriction::FORBID};
  ASSERT_TRUE(nopartition->satisfies_restrictions(restrictions2));
}

TEST_F(NoPartitionTest, LaunchDomain)
{
  ASSERT_FALSE(nopartition->has_launch_domain());
  ASSERT_THROW(static_cast<void>(nopartition->launch_domain()), std::invalid_argument);
}

TEST_F(NoPartitionTest, ToString)
{
  constexpr std::string_view nopartition_str = "NoPartition";

  std::stringstream ss;
  ss << *nopartition;

  ASSERT_EQ(ss.str(), nopartition_str);
  ASSERT_EQ(nopartition->to_string(), nopartition_str);
}

TEST_F(NoPartitionTest, Scale)
{
  constexpr std::uint64_t factors[] = {2, 2};
  auto scaled                       = nopartition->scale(factors);
  ASSERT_EQ(scaled->kind(), legate::detail::Partition::Kind::NO_PARTITION);
}

TEST_F(NoPartitionTest, Bloat)
{
  constexpr std::uint64_t low_offsets[]  = {0, 0};
  constexpr std::uint64_t high_offsets[] = {1, 1};
  auto bloated                           = nopartition->bloat(low_offsets, high_offsets);
  ASSERT_EQ(bloated->kind(), legate::detail::Partition::Kind::NO_PARTITION);
}

TEST_F(NoPartitionTest, Construct)
{
  auto constructed = nopartition->construct(Legion::LogicalRegion{}, /*complete=*/false);
  ASSERT_EQ(constructed, Legion::LogicalPartition::NO_PART);
}

TEST_F(NoPartitionTest, Convert)
{
  auto nopartion1     = nopartition->convert(nopartition, nullptr);
  auto nopartion1_raw = nopartion1.get();
  auto nopartion_raw  = nopartition.get();
  ASSERT_EQ(nopartion1_raw, nopartion_raw);
}

TEST_F(NoPartitionTest, Invert)
{
  auto nopartion1     = nopartition->invert(nopartition, nullptr);
  auto nopartion1_raw = nopartion1.get();
  auto nopartion_raw  = nopartition.get();
  ASSERT_EQ(nopartion1_raw, nopartion_raw);
}

TEST_F(NoPartitionTest, IsDisjointFor)
{
  auto domain1 = Legion::Domain::NO_DOMAIN;
  ASSERT_TRUE(nopartition->is_disjoint_for(domain1));

  auto domain2 = Legion::Domain{Legion::Rect<1>{0, 0}};
  ASSERT_TRUE(nopartition->is_disjoint_for(domain2));

  auto domain3 = Legion::Domain{Legion::Rect<1>{0, 1}};
  ASSERT_FALSE(nopartition->is_disjoint_for(domain3));
}

TEST_F(NoPartitionTest, ColorShape)
{
  ASSERT_THROW(static_cast<void>(nopartition->color_shape()), std::invalid_argument);
}

}  // namespace unit
