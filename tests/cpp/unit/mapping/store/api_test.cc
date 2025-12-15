/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/store.h>

#include <gtest/gtest.h>

#include <unit/mapping/utils.h>
#include <utilities/utilities.h>

namespace mapping_store_test {

using mapping_utils_test::create_test_store;

namespace {

using MappingStoreApiTest = DefaultFixture;

}  // namespace

TEST_F(MappingStoreApiTest, IsFuture)
{
  auto detail_store = create_test_store(legate::Shape{8}, legate::int32());
  const legate::mapping::Store store{&detail_store};

  ASSERT_TRUE(store.is_future());
}

TEST_F(MappingStoreApiTest, Unbound)
{
  auto detail_store = create_test_store(legate::Shape{7}, legate::int32());
  const legate::mapping::Store store{&detail_store};

  ASSERT_FALSE(store.unbound());
}

TEST_F(MappingStoreApiTest, Dim)
{
  auto detail_store_1d = create_test_store(legate::Shape{1}, legate::int32());
  auto detail_store_2d = create_test_store(legate::Shape{5, 6}, legate::float32());
  auto detail_store_3d = create_test_store(legate::Shape{2, 3, 4}, legate::int64());
  const legate::mapping::Store store_1d{&detail_store_1d};
  const legate::mapping::Store store_2d{&detail_store_2d};
  const legate::mapping::Store store_3d{&detail_store_3d};

  ASSERT_EQ(store_1d.dim(), 1);
  ASSERT_EQ(store_2d.dim(), 2);
  ASSERT_EQ(store_3d.dim(), 3);
}

TEST_F(MappingStoreApiTest, IsReduction)
{
  auto detail_store = create_test_store(legate::Shape{8}, legate::int32());
  const legate::mapping::Store store{&detail_store};

  ASSERT_FALSE(store.is_reduction());
}

TEST_F(MappingStoreApiTest, Redop)
{
  auto detail_store = create_test_store(legate::Shape{7}, legate::int32());
  const legate::mapping::Store store{&detail_store};

  ASSERT_EQ(store.redop(), legate::GlobalRedopID{-1});
}

TEST_F(MappingStoreApiTest, CanColocateWith)
{
  auto detail_store1 = create_test_store(legate::Shape{5}, legate::int32());
  auto detail_store2 = create_test_store(legate::Shape{9}, legate::float32());
  const legate::mapping::Store store1{&detail_store1};
  const legate::mapping::Store store2{&detail_store2};

  // Future stores cannot colocate
  ASSERT_FALSE(store1.can_colocate_with(store2));
  ASSERT_FALSE(store2.can_colocate_with(store1));
  ASSERT_FALSE(store1.can_colocate_with(store1));
}

TEST_F(MappingStoreApiTest, Domain)
{
  constexpr std::int32_t SHAPE_SIZE = 20;
  auto detail_store                 = create_test_store(legate::Shape{SHAPE_SIZE}, legate::int32());
  const legate::mapping::Store store{&detail_store};
  auto domain = store.domain();

  ASSERT_EQ(domain.dim, 1);
  ASSERT_EQ(domain.lo()[0], 0);
  ASSERT_EQ(domain.hi()[0], SHAPE_SIZE - 1);
}

TEST_F(MappingStoreApiTest, Shape)
{
  constexpr std::int32_t SHAPE_SIZE = 39;
  auto detail_store                 = create_test_store(legate::Shape{SHAPE_SIZE}, legate::int32());
  const legate::mapping::Store store{&detail_store};
  auto shape = store.shape<1>();

  ASSERT_EQ(shape.lo[0], 0);
  ASSERT_EQ(shape.hi[0], SHAPE_SIZE - 1);
}

TEST_F(MappingStoreApiTest, Impl)
{
  auto detail_store = create_test_store(legate::Shape{2}, legate::int32());
  const legate::mapping::Store store{&detail_store};
  const auto* impl = store.impl();

  ASSERT_NE(impl, nullptr);
  ASSERT_TRUE(impl->is_future());
  ASSERT_EQ(impl->dim(), 1);
}

}  // namespace mapping_store_test
