/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/transform/delinearize.h>
#include <legate/data/detail/transform/non_invertible_transformation.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/runtime/detail/runtime.h>

#include <legion/api/data.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace unit {

class OpaqueTest : public DefaultFixture {
 public:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    opaque = create_opaque();
  }

  [[nodiscard]] legate::InternalSharedPtr<legate::detail::Opaque> create_opaque()
  {
    auto runtime      = legate::Runtime::get_runtime();
    auto runtime_impl = runtime->impl();

    // For Opaque partition the name of the IndexSpace and IndexPartition is good enough.
    // The structures don't necessarily have to be filled (they should be populated as
    // the task using the partition is completing execution) but we don't have a way to
    // only get the name of them so we have to form proper filled out data structures.
    launch_domain    = Legion::Domain{Legion::Rect<1>{0, 1}};
    launch_is        = runtime_impl->find_or_create_index_space(launch_domain);
    auto data_domain = Legion::Domain{Legion::Rect<1>{0, 3}};
    auto& data_is    = runtime_impl->find_or_create_index_space(data_domain);
    partition        = runtime_impl->create_equal_partition(data_is, launch_is);

    return legate::detail::create_opaque(data_is, partition, launch_domain);
  }

  legate::InternalSharedPtr<legate::detail::Opaque> opaque;
  Legion::IndexSpace launch_is;
  Legion::Domain launch_domain;
  Legion::IndexPartition partition;
};

TEST_F(OpaqueTest, Kind) { ASSERT_EQ(opaque->kind(), legate::detail::Partition::Kind::OPAQUE); }

TEST_F(OpaqueTest, Compare)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto runtime_impl = runtime->impl();

  auto data_domain = Legion::Domain{Legion::Rect<1>{0, 5}};
  auto& data_is    = runtime_impl->find_or_create_index_space(data_domain);

  auto opaque1 = legate::detail::create_opaque(data_is, partition, launch_domain);

  ASSERT_TRUE(*opaque1 == *opaque);
  ASSERT_FALSE(*opaque < *opaque1);
}

TEST_F(OpaqueTest, ColorShape) { ASSERT_THAT(opaque->color_shape(), ::testing::ElementsAre(2)); }

TEST_F(OpaqueTest, IsCompleteFor)
{
  // This store itself does not actually matter, we just need to create one in order to get a
  // `Storage` reference.
  const auto store =
    legate::Runtime::get_runtime()->create_store(legate::Shape{1}, legate::int32());

  ASSERT_TRUE(opaque->is_complete_for(*store.impl()->get_storage()));
}

TEST_F(OpaqueTest, IsConvertible) { ASSERT_FALSE(opaque->is_convertible()); }

TEST_F(OpaqueTest, LaunchDomain)
{
  ASSERT_TRUE(opaque->has_launch_domain());

  auto domain = Legion::Domain{Legion::Rect<1>{0, 1}};
  ASSERT_EQ(opaque->launch_domain(), domain);
}

TEST_F(OpaqueTest, IsDisjointFor)
{
  auto domain1 = Legion::Domain::NO_DOMAIN;
  ASSERT_TRUE(opaque->is_disjoint_for(domain1));

  auto domain2 = Legion::Domain{Legion::Rect<1>{0, 1}};
  ASSERT_TRUE(opaque->is_disjoint_for(domain2));

  auto domain3 = Legion::Domain{Legion::Rect<1>{0, 2}};
  ASSERT_FALSE(opaque->is_disjoint_for(domain3));
}

TEST_F(OpaqueTest, SatisfiesRestrictions)
{
  auto restrictions1 =
    legate::detail::Restrictions{legate::detail::SmallVector{legate::detail::Restriction::ALLOW}};
  ASSERT_TRUE(restrictions1.are_satisfied_by(*opaque));

  auto restrictions2 =
    legate::detail::Restrictions{legate::detail::SmallVector{legate::detail::Restriction::AVOID}};
  ASSERT_TRUE(restrictions2.are_satisfied_by(*opaque));

  auto restrictions3 =
    legate::detail::Restrictions{legate::detail::SmallVector{legate::detail::Restriction::FORBID}};
  ASSERT_FALSE(restrictions3.are_satisfied_by(*opaque));
}

TEST_F(OpaqueTest, SatisfiesRestrictionsNegative)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Sizes are only checked in debug builds";
  }

  auto restrictions1 = legate::detail::Restrictions{legate::detail::SmallVector{
    legate::detail::Restriction::ALLOW, legate::detail::Restriction::AVOID}};
  ASSERT_THAT([&] { static_cast<void>(restrictions1.are_satisfied_by(*opaque)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Arguments to zip_equal() are not all equal")));

  auto restrictions2 = legate::detail::Restrictions{};
  ASSERT_THAT([&] { static_cast<void>(restrictions2.are_satisfied_by(*opaque)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Arguments to zip_equal() are not all equal")));
}

TEST_F(OpaqueTest, Scale)
{
  // Scale is not implemented for opaque partitions
  constexpr std::uint64_t factors[] = {2, 2};
  ASSERT_THAT(
    [&] { static_cast<void>(opaque->scale(factors)); },
    ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Not implemented")));
}

TEST_F(OpaqueTest, Bloat)
{
  // Bloat is not implemented for opaque partitions
  constexpr std::uint64_t low_offsets[]  = {0, 0};
  constexpr std::uint64_t high_offsets[] = {1, 1};
  ASSERT_THAT(
    [&] { static_cast<void>(opaque->bloat(low_offsets, high_offsets)); },
    ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Not implemented")));
}

TEST_F(OpaqueTest, Convert)
{
  // Test identity transformation
  auto transform1  = legate::make_internal_shared<legate::detail::TransformStack>();
  auto opaque1     = opaque->convert(opaque, transform1);
  auto opaque1_raw = opaque1.get();
  auto opaque_raw  = opaque.get();
  ASSERT_EQ(opaque1_raw, opaque_raw);

  // Test non-identity transformation
  auto dim        = 2;
  auto sizes      = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{2, 1};
  auto transform2 = legate::make_internal_shared<legate::detail::TransformStack>(
    std::make_unique<legate::detail::Delinearize>(dim, std::move(sizes)), std::move(transform1));
  ASSERT_THAT([&] { static_cast<void>(opaque->convert(opaque, transform2)); },
              ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
                "We can not convert an Opaque Partition without identity transformation")));
}

TEST_F(OpaqueTest, Invert)
{
  // Test identity transformation
  auto transform1  = legate::make_internal_shared<legate::detail::TransformStack>();
  auto opaque1     = opaque->invert(opaque, transform1);
  auto opaque1_raw = opaque1.get();
  auto opaque_raw  = opaque.get();
  ASSERT_EQ(opaque1_raw, opaque_raw);

  // Test non-identity transformation
  auto dim        = 1;
  auto sizes      = legate::detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{1};
  auto transform2 = legate::make_internal_shared<legate::detail::TransformStack>(
    std::make_unique<legate::detail::Delinearize>(dim, std::move(sizes)), std::move(transform1));

  ASSERT_THAT([&] { static_cast<void>(opaque->invert(opaque, transform2)); },
              ::testing::ThrowsMessage<legate::detail::NonInvertibleTransformation>(
                ::testing::HasSubstr("Non-invertible transformation")));
}

TEST_F(OpaqueTest, ToString)
{
  const std::string opaque_str =
    fmt::format("Opaque(IndexPartition({}, {}))", partition.get_id(), partition.get_tree_id());

  ASSERT_EQ(opaque->to_string(), opaque_str);
}

}  // namespace unit
