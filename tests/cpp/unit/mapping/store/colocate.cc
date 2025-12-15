/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/store.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace mapping_store_test {

using legate::mapping::detail::FutureWrapper;
using legate::mapping::detail::Store;

namespace {

using MappingStoreColocateTest = DefaultFixture;

}  // namespace

TEST_F(MappingStoreColocateTest, FutureStores)
{
  const legate::InternalSharedPtr<legate::detail::Type> type1{legate::int32().impl()};
  const legate::InternalSharedPtr<legate::detail::Type> type2{legate::float32().impl()};
  const Legion::Domain domain1{Legion::Rect<1>{0, 9}};
  const Legion::Domain domain2{Legion::Rect<1>{0, 3}};
  const FutureWrapper future1{/*idx=*/0, domain1};
  const FutureWrapper future2{/*idx=*/1, domain2};
  const Store store1{/*dim=*/1, type1, future1};
  const Store store2{/*dim=*/1, type2, future2};

  // Future stores cannot colocate with anything
  ASSERT_FALSE(store1.can_colocate_with(store2));
  ASSERT_FALSE(store2.can_colocate_with(store1));
}

TEST_F(MappingStoreColocateTest, FutureWithItself)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain{Legion::Rect<1>{0, 9}};
  const FutureWrapper future{/*idx=*/0, domain};
  const Store store{/*dim=*/1, type, future};

  // Even with itself, future cannot colocate
  ASSERT_FALSE(store.can_colocate_with(store));
}

TEST_F(MappingStoreColocateTest, FutureStoresDifferentDims)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain1d{Legion::Rect<1>{0, 9}};
  const Legion::Domain domain2d{Legion::Rect<2>{{0, 0}, {4, 4}}};
  const FutureWrapper future1d{/*idx=*/0, domain1d};
  const FutureWrapper future2d{/*idx=*/1, domain2d};
  const Store store1d{/*dim=*/1, type, future1d};
  const Store store2d{/*dim=*/2, type, future2d};

  // Future stores of different dimensions still cannot colocate
  ASSERT_FALSE(store1d.can_colocate_with(store2d));
  ASSERT_FALSE(store2d.can_colocate_with(store1d));
}

TEST_F(MappingStoreColocateTest, FutureStoresSameIndexDifferentDomain)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain1{Legion::Rect<1>{0, 9}};
  const Legion::Domain domain2{Legion::Rect<1>{0, 19}};

  // Same future index but different domains
  const FutureWrapper future1{/*idx=*/0, domain1};
  const FutureWrapper future2{/*idx=*/0, domain2};
  const Store store1{/*dim=*/1, type, future1};
  const Store store2{/*dim=*/1, type, future2};

  // Future stores cannot colocate even with same index
  ASSERT_FALSE(store1.can_colocate_with(store2));
  ASSERT_FALSE(store2.can_colocate_with(store1));
}

}  // namespace mapping_store_test
