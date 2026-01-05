/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/transform_stack.h>
#include <legate/mapping/detail/store.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace mapping_store_test {

using legate::mapping::detail::FutureWrapper;
using legate::mapping::detail::Store;

namespace {

using MappingStoreDomainTest = DefaultFixture;

}  // namespace

TEST_F(MappingStoreDomainTest, FutureDomain)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int64().impl()};
  const Legion::Domain expected_domain{Legion::Rect<1>{5, 8}};
  const FutureWrapper future{/*idx=*/0, expected_domain};
  auto transform             = legate::make_internal_shared<legate::detail::TransformStack>();
  constexpr std::int32_t DIM = 1;
  const Store store{DIM, type, future, std::move(transform)};
  auto domain = store.domain();

  ASSERT_EQ(domain.dim, DIM);
  ASSERT_EQ(domain.lo()[0], 5);
  ASSERT_EQ(domain.hi()[0], 8);
}

TEST_F(MappingStoreDomainTest, EmptyDomain)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain empty_domain{Legion::Rect<1>{0, -1}};  // Empty domain
  const FutureWrapper future{/*idx=*/0, empty_domain};
  auto transform             = legate::make_internal_shared<legate::detail::TransformStack>();
  constexpr std::int32_t DIM = 1;
  const Store store{DIM, type, future, std::move(transform)};

  ASSERT_TRUE(store.is_future());
  ASSERT_TRUE(store.valid());
  ASSERT_EQ(store.dim(), DIM);

  auto domain = store.domain();

  ASSERT_EQ(domain.dim, DIM);
  ASSERT_TRUE(domain.empty());
}

TEST_F(MappingStoreDomainTest, LargeDomain)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  constexpr std::int64_t MAX_COORD = 999999;
  const Legion::Domain large_domain{Legion::Rect<1>{0, MAX_COORD}};
  const FutureWrapper future{/*idx=*/0, large_domain};
  auto transform             = legate::make_internal_shared<legate::detail::TransformStack>();
  constexpr std::int32_t DIM = 1;
  const Store store{DIM, type, future, std::move(transform)};
  auto domain = store.domain();

  ASSERT_EQ(domain.dim, DIM);
  ASSERT_EQ(domain.get_volume(), MAX_COORD + 1);
}

TEST_F(MappingStoreDomainTest, NulloptTransform)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain expected_domain{Legion::Rect<1>{0, 9}};
  const FutureWrapper future{/*idx=*/0, expected_domain};
  constexpr std::int32_t DIM = 1;
  const Store store{DIM, type, future, std::nullopt};
  auto domain = store.domain();

  ASSERT_EQ(domain.dim, DIM);
  ASSERT_EQ(domain.lo()[0], 0);
  ASSERT_EQ(domain.hi()[0], 9);
}

}  // namespace mapping_store_test
