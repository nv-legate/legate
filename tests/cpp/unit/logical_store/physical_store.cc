/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace logical_store_physical_test {

using LogicalStorePhysicalUnit = DefaultFixture;

TEST_F(LogicalStorePhysicalUnit, BoundStore)
{
  auto runtime        = legate::Runtime::get_runtime();
  auto store          = runtime->create_store(legate::Shape{4, 4}, legate::int64());
  auto physical_store = store.get_physical_store();

  ASSERT_FALSE(physical_store.is_unbound_store());
  ASSERT_FALSE(physical_store.is_future());

  constexpr std::uint32_t DIM = 2;

  ASSERT_EQ(physical_store.dim(), DIM);
  ASSERT_EQ(physical_store.shape<DIM>(), (legate::Rect<DIM>{{0, 0}, {3, 3}}));
  ASSERT_TRUE(physical_store.valid());
  ASSERT_EQ(physical_store.code(), legate::Type::Code::INT64);
}

TEST_F(LogicalStorePhysicalUnit, EmptyShapeBoundStore)
{
  auto runtime        = legate::Runtime::get_runtime();
  auto store          = runtime->create_store(legate::Shape{}, legate::int64());
  auto physical_store = store.get_physical_store();

  ASSERT_FALSE(physical_store.is_unbound_store());
  ASSERT_FALSE(physical_store.is_future());
  ASSERT_EQ(physical_store.dim(), 0);

  constexpr std::uint32_t DIM = 1;

  // shape of 0-D store
  ASSERT_EQ(physical_store.shape<DIM>(), (legate::Rect<DIM>{0, 0}));
  ASSERT_TRUE(physical_store.valid());
  ASSERT_EQ(physical_store.code(), legate::Type::Code::INT64);
}

TEST_F(LogicalStorePhysicalUnit, UnboundStore)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::int64(), LEGATE_MAX_DIM);

  ASSERT_THROW(static_cast<void>(store.get_physical_store()), std::invalid_argument);
}

TEST_F(LogicalStorePhysicalUnit, ScalarStore)
{
  auto runtime                        = legate::Runtime::get_runtime();
  constexpr std::int32_t SCALAR_VALUE = 10;
  auto store                          = runtime->create_store(legate::Scalar{SCALAR_VALUE});
  auto physical_store                 = store.get_physical_store();

  ASSERT_FALSE(physical_store.is_unbound_store());
  ASSERT_TRUE(physical_store.is_future());

  static constexpr std::uint32_t DIM = 1;

  ASSERT_EQ(physical_store.dim(), DIM);
  ASSERT_EQ(physical_store.shape<DIM>(), (legate::Rect<DIM>{0, 0}));
  ASSERT_TRUE(physical_store.valid());
  ASSERT_EQ(physical_store.code(), legate::Type::Code::INT32);
}

}  // namespace logical_store_physical_test
