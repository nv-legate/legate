/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/allocation_cache.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utilities/utilities.h>
#include <vector>

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

TEST_F(CpuAllocationCacheTest, FlushCache)
{
  // Cache can hold up to 8 bytes. Fill it to capacity, then returning another allocation
  // exceeds the limit and triggers a full cache flush.
  const std::uint64_t MAX_SIZE   = 8;
  const std::uint64_t ALLOC_SIZE = 4;
  legate::detail::CpuAllocationCache cache{MAX_SIZE};
  const void* alloc1 = cache.get_allocation(ALLOC_SIZE);
  const void* alloc2 = cache.get_allocation(ALLOC_SIZE);
  const void* alloc3 = cache.get_allocation(ALLOC_SIZE);

  cache.return_allocation(alloc1, ALLOC_SIZE);
  cache.return_allocation(alloc2, ALLOC_SIZE);
  // Cache now holds 8 bytes (at capacity). Returning alloc3 would make it 12 > 8, so the
  // cache flushes everything and deletes alloc3.
  cache.return_allocation(alloc3, ALLOC_SIZE);

  // After flush the cache is empty and functional again. Return a new allocation and
  // retrieve it -- the cache should hand back the same pointer, proving it was reset.
  const void* alloc4 = cache.get_allocation(ALLOC_SIZE);

  cache.return_allocation(alloc4, ALLOC_SIZE);

  const void* alloc5 = cache.get_allocation(ALLOC_SIZE);

  ASSERT_EQ(alloc4, alloc5);
  cache.return_allocation(alloc5, ALLOC_SIZE);
}

TEST_F(CpuAllocationCacheTest, ExceedsMaxBinCount)
{
  const std::uint64_t MAX_SIZE   = 1024;
  const std::uint64_t ALLOC_SIZE = 4;
  constexpr auto COUNT           = legate::detail::AllocationCache::MAX_BIN_COUNT + 1;
  legate::detail::CpuAllocationCache cache{MAX_SIZE};
  std::vector<const void*> allocs;

  allocs.reserve(COUNT);
  for (std::size_t i = 0; i < COUNT; ++i) {
    allocs.push_back(cache.get_allocation(ALLOC_SIZE));
  }
  for (std::size_t i = 0; i < COUNT - 1; ++i) {
    cache.return_allocation(allocs[i], ALLOC_SIZE);
  }

  // 11th return: bin already has 10, calls deleter_ to delete the allocation.
  cache.return_allocation(allocs[COUNT - 1], ALLOC_SIZE);

  // The 10 cached allocations are still available (stack order: last returned first out).
  const void* retrieved = cache.get_allocation(ALLOC_SIZE);

  ASSERT_EQ(retrieved, allocs[COUNT - 2]);
  cache.return_allocation(retrieved, ALLOC_SIZE);
}

TEST_F(CpuAllocationCacheTest, SmallAllocationIsNoop)
{
  constexpr std::size_t SMALL_SIZE = 1024;
  constexpr int FILL_BYTE          = 0xAB;
  auto buf                         = std::make_unique<std::byte[]>(SMALL_SIZE);

  std::memset(buf.get(), FILL_BYTE, SMALL_SIZE);
  legate::detail::maybe_advise_huge_pages(buf.get(), SMALL_SIZE);

  ASSERT_TRUE(std::all_of(
    buf.get(), buf.get() + SMALL_SIZE, [](std::byte b) { return b == std::byte{FILL_BYTE}; }));
}

TEST_F(CpuAllocationCacheTest, LargeAllocationAdvisesHugePages)
{
  constexpr std::size_t FOUR_MB = 4U * (1U << 20U);
  constexpr int FILL_BYTE       = 0xCD;
  auto buf                      = std::make_unique<std::byte[]>(FOUR_MB);

  std::memset(buf.get(), FILL_BYTE, FOUR_MB);
  legate::detail::maybe_advise_huge_pages(buf.get(), FOUR_MB);

  ASSERT_TRUE(std::all_of(
    buf.get(), buf.get() + FOUR_MB, [](std::byte b) { return b == std::byte{FILL_BYTE}; }));
}

}  // namespace allocation_cache_test
