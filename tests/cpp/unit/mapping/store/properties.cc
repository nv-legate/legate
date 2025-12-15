/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/promote.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/mapping/detail/store.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace mapping_store_test {

using legate::mapping::detail::FutureWrapper;
using legate::mapping::detail::Store;

namespace {

using MappingStorePropertiesTest = DefaultFixture;

}  // namespace

TEST_F(MappingStorePropertiesTest, DefaultConstruction)
{
  const Store store;

  ASSERT_FALSE(store.is_future());
  ASSERT_FALSE(store.unbound());
  ASSERT_EQ(store.dim(), -1);
  ASSERT_FALSE(store.type());
  ASSERT_FALSE(store.is_reduction());
  ASSERT_EQ(store.redop(), legate::GlobalRedopID{-1});
}

TEST_F(MappingStorePropertiesTest, FutureConstruction)
{
  constexpr std::int32_t dim = 2;
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain{Legion::Rect<2>{{0, 0}, {9, 9}}};
  const FutureWrapper future{/*idx=*/0, domain};
  const Store store{dim, type, future};

  ASSERT_TRUE(store.is_future());
  ASSERT_FALSE(store.unbound());
  ASSERT_EQ(store.dim(), dim);
  ASSERT_TRUE(store.type());
  ASSERT_EQ(store.type()->code, legate::int32().code());
  ASSERT_FALSE(store.is_reduction());
  ASSERT_TRUE(store.valid());
}

TEST_F(MappingStorePropertiesTest, FutureAccessors)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain{Legion::Rect<1>{0, 9}};
  const FutureWrapper future{/*idx=*/5, domain};
  const Store store{/*dim=*/1, type, future};

  ASSERT_TRUE(store.is_future());
  ASSERT_NO_THROW({ static_cast<void>(store.future()); });
  ASSERT_EQ(store.future().index(), 5);
  ASSERT_EQ(store.future_index(), 5);
}

TEST_F(MappingStorePropertiesTest, ValidFuture)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain{Legion::Rect<1>{0, 9}};
  const FutureWrapper future{/*idx=*/0, domain};
  const Store store{/*dim=*/1, type, future};

  ASSERT_TRUE(store.valid());
}

TEST_F(MappingStorePropertiesTest, NotTransformed)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain{Legion::Rect<1>{0, 9}};
  const FutureWrapper future{/*idx=*/0, domain};

  // Without transform
  const Store store_no_transform{/*dim=*/1, type, future};
  ASSERT_FALSE(store_no_transform.transformed());

  // With identity transform
  auto identity_transform = legate::make_internal_shared<legate::detail::TransformStack>();

  const Store store_identity{/*dim=*/1, type, future, std::move(identity_transform)};
  ASSERT_FALSE(store_identity.transformed());
}

TEST_F(MappingStorePropertiesTest, Transformed)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain{Legion::Rect<1>{0, 9}};
  const FutureWrapper future{/*idx=*/0, domain};

  // Create a non-identity transform (Promote adds a dimension)
  auto parent = legate::make_internal_shared<legate::detail::TransformStack>();
  auto promote_transform =
    std::make_unique<legate::detail::Promote>(/*extra_dim=*/1, /*dim_size=*/8);
  auto transform = legate::make_internal_shared<legate::detail::TransformStack>(
    std::move(promote_transform), parent);

  const Store store{/*dim=*/1, type, future, std::move(transform)};
  ASSERT_TRUE(store.transformed());
}

TEST_F(MappingStorePropertiesTest, Reduction)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain{Legion::Rect<1>{0, 9}};
  const FutureWrapper future{/*idx=*/0, domain};
  const Store store{/*dim=*/1, type, future};

  ASSERT_FALSE(store.is_reduction());
  ASSERT_EQ(store.redop(), legate::GlobalRedopID{-1});
}

}  // namespace mapping_store_test
