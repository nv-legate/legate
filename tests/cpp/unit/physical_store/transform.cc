/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace physical_store_transform_test {

using PhysicalStoreTransformUnit = DefaultFixture;

TEST_F(PhysicalStoreTransformUnit, FutureStore)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{1});
  auto store         = logical_store.get_physical_store();

  ASSERT_FALSE(store.transformed());

  auto promoted = logical_store.promote(/*extra_dim=*/0, /*dim_size=*/1);
  store         = promoted.get_physical_store();

  ASSERT_TRUE(store.transformed());
}

TEST_F(PhysicalStoreTransformUnit, BoundStore)
{
  auto runtime                  = legate::Runtime::get_runtime();
  static constexpr auto EXTENTS = 7;
  auto logical_store            = runtime->create_store({0, EXTENTS}, legate::int64());
  auto store                    = logical_store.get_physical_store();

  ASSERT_FALSE(store.transformed());

  auto promoted = logical_store.promote(/*extra_dim=*/0, /*dim_size=*/1);
  store         = promoted.get_physical_store();

  ASSERT_TRUE(store.transformed());
}

}  // namespace physical_store_transform_test
