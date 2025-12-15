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

namespace {

using MappingStoreFutureWrapperTest = DefaultFixture;

}  // namespace

TEST_F(MappingStoreFutureWrapperTest, DefaultConstruction)
{
  const FutureWrapper future;

  ASSERT_EQ(future.index(), static_cast<std::uint32_t>(-1));
  ASSERT_EQ(future.dim(), 0);
}

TEST_F(MappingStoreFutureWrapperTest, Construction)
{
  constexpr std::uint32_t idx = 42;
  const Legion::Domain domain{Legion::Rect<2>{{0, 0}, {9, 9}}};

  const FutureWrapper future{idx, domain};

  ASSERT_EQ(future.index(), idx);
  ASSERT_EQ(future.dim(), 2);
  ASSERT_EQ(future.domain().dim, 2);
}

}  // namespace mapping_store_test
