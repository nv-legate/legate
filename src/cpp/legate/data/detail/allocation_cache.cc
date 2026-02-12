/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/allocation_cache.h>

#include <legate_defines.h>

#include <legate/utilities/abort.h>
#include <legate/utilities/macros.h>

#include <cstring>
#include <unordered_map>

#if LEGATE_DEFINED(LEGATE_LINUX)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace legate::detail {

AllocationCache::~AllocationCache() { flush_free_allocations_(); }

void* AllocationCache::get_allocation(std::uint64_t size)
{
  auto it = allocations_.find(size);

  if (it == allocations_.end() || it->second.empty()) {
    return allocator_(size);
  }

  // Use an existing allocation from the cache.
  const auto result = it->second.top();

  it->second.pop();
  total_size_ -= size;
  return result;
}

void AllocationCache::return_allocation(const void* ptr, std::uint64_t size)
{
  if (total_size_ + size > max_total_size_) {
    deleter_(const_cast<void*>(ptr));

    // empty cache as it is full
    flush_free_allocations_();
    return;
  }

  const auto [it, found] = allocations_.try_emplace(size, SizedAllocationStack{});

  if (it->second.size() < MAX_BIN_COUNT) {
    it->second.push(const_cast<void*>(ptr));
    total_size_ += size;
  } else {
    deleter_(const_cast<void*>(ptr));
  }
}

void AllocationCache::flush_free_allocations_()
{
  for (auto&& [size, sized_allocs] : allocations_) {
    while (!sized_allocs.empty()) {
      const auto ptr = sized_allocs.top();

      sized_allocs.pop();
      deleter_(ptr);
    }
  }
  total_size_ = 0;
  allocations_.clear();
}

CpuAllocationCache::CpuAllocationCache(std::uint64_t max_size,
                                       std::optional<std::uint8_t> alignment) noexcept
  : AllocationCache{[alignment_opt = alignment](std::uint64_t size) {
                      if (alignment_opt.has_value()) {
                        return new (std::align_val_t{*alignment_opt}, std::nothrow) std::byte[size];
                      }
                      return new (std::nothrow) std::byte[size];
                    },
                    [alignment_opt = alignment](void* ptr) {
                      if (alignment_opt.has_value()) {
                        ::operator delete[](static_cast<std::byte*>(ptr),
                                            std::align_val_t{*alignment_opt});
                      } else {
                        delete[] static_cast<std::byte*>(ptr);
                      }
                    },
                    max_size}
{
}

void maybe_advise_huge_pages([[maybe_unused]] const void* ptr, [[maybe_unused]] std::size_t size)
{
#if LEGATE_DEFINED(LEGATE_LINUX)
  constexpr std::size_t MB            = 1U << 20U;
  constexpr std::uint8_t MAX_ATTEMPTS = 3;

  if (size < 4U * MB) {
    return;
  }

  // Attempt to advise the OS to use huge pages for the given allocation.
  const std::size_t page_size = getpagesize();
  const std::uintptr_t offset = page_size - (reinterpret_cast<std::uintptr_t>(ptr) % page_size);
  const std::uint64_t length  = size - offset;
  const void* new_ptr =
    reinterpret_cast<const void*>(reinterpret_cast<const std::byte*>(ptr) + offset);

  for (std::uint8_t attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
    const int ret = madvise(const_cast<void*>(new_ptr), length, MADV_HUGEPAGE);

    if (ret == 0) {
      return;
    }

    if (const auto err = errno; err != EAGAIN) {
      LEGATE_ABORT(std::strerror(err));
    }
  }
#endif
}

}  // namespace legate::detail
