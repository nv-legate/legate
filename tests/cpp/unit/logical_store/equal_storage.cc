/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace logical_store_equal_storage_test {

using LogicalStoreEqualStorageUnit = DefaultFixture;

constexpr std::int32_t SCALAR_VALUE = 10;

TEST_F(LogicalStoreEqualStorageUnit, Self)
{
  const auto runtime = legate::Runtime::get_runtime();
  const auto store   = runtime->create_store(legate::Scalar{SCALAR_VALUE});

  ASSERT_TRUE(store.equal_storage(store));
}

TEST_F(LogicalStoreEqualStorageUnit, NotEqual)
{
  const auto runtime = legate::Runtime::get_runtime();
  const auto store   = runtime->create_store(legate::Scalar{SCALAR_VALUE});
  const auto store2  = runtime->create_store(legate::Scalar{SCALAR_VALUE});

  // Unrelated stores are in fact
  ASSERT_FALSE(store.equal_storage(store2));
  ASSERT_FALSE(store2.equal_storage(store));
}

TEST_F(LogicalStoreEqualStorageUnit, Sliced)
{
  const auto runtime = legate::Runtime::get_runtime();
  /// [Store::equal_storage: Comparing sliced stores]
  const auto store       = runtime->create_store(legate::Shape{4, 3}, legate::int64());
  const auto transformed = store.slice(1, legate::Slice{-2, -1});

  // Slices partition a store into a parent and sub-store which both cover distinct regions,
  // and hence don't share storage.
  ASSERT_FALSE(store.equal_storage(transformed));
  /// [Store::equal_storage: Comparing sliced stores]
  ASSERT_FALSE(transformed.equal_storage(store));
}

TEST_F(LogicalStoreEqualStorageUnit, Transpoe)
{
  const auto runtime = legate::Runtime::get_runtime();
  /// [Store::equal_storage: Comparing transposed stores]
  const auto store       = runtime->create_store(legate::Shape{4, 3}, legate::int64());
  const auto transformed = store.transpose({1, 0});

  // Transposing a store doesn't modify the storage
  ASSERT_TRUE(store.equal_storage(transformed));
  /// [Store::equal_storage: Comparing transposed stores]
  ASSERT_TRUE(transformed.equal_storage(store));

  const auto transformed2 = transformed.transpose({1, 0});

  ASSERT_TRUE(transformed.equal_storage(transformed2));
  ASSERT_TRUE(transformed2.equal_storage(transformed));
  ASSERT_TRUE(store.equal_storage(transformed2));
  ASSERT_TRUE(transformed2.equal_storage(store));
}

TEST_F(LogicalStoreEqualStorageUnit, Future)
{
  const auto runtime  = legate::Runtime::get_runtime();
  const auto store    = runtime->create_store(legate::Scalar{SCALAR_VALUE});
  const auto promoted = store.promote(0, 5);

  ASSERT_TRUE(promoted.get_physical_store().is_future());
  ASSERT_TRUE(promoted.equal_storage(store));
  ASSERT_TRUE(store.equal_storage(promoted));
}

}  // namespace logical_store_equal_storage_test
