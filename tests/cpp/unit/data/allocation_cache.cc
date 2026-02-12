/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/allocation_cache.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <utilities/utilities.h>

namespace allocation_cache_test {

using CpuAllocationCacheTest = DefaultFixture;

TEST_F(CpuAllocationCacheTest, SimpleCache)
{
  const std::uint64_t MAX_SIZE   = 8;
  const std::uint64_t ALLOC_SIZE = 4;

  legate::detail::CpuAllocationCache cache{MAX_SIZE};
  const void* alloc1 = cache.get_allocation(ALLOC_SIZE);

  cache.return_allocation(alloc1, ALLOC_SIZE);

  const void* alloc2 = cache.get_allocation(ALLOC_SIZE);

  ASSERT_EQ(alloc1, alloc2);

  cache.return_allocation(const_cast<void*>(alloc2), ALLOC_SIZE);
}

TEST_F(CpuAllocationCacheTest, Alignment)
{
  const std::uint64_t MAX_SIZE   = 8;
  const std::uint64_t ALLOC_SIZE = 4;
  const std::uint64_t ALIGNMENT  = 32;

  legate::detail::CpuAllocationCache cache{MAX_SIZE, ALIGNMENT};
  const void* alloc = cache.get_allocation(ALLOC_SIZE);
  const auto addr   = reinterpret_cast<std::uintptr_t>(alloc);

  ASSERT_EQ(addr % ALIGNMENT, 0);

  cache.return_allocation(alloc, ALLOC_SIZE);
}

}  // namespace allocation_cache_test
