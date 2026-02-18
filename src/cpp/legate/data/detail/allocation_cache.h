/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/inplace_vector>

#include <cstdint>
#include <functional>
#include <optional>
#include <stack>
#include <unordered_map>

namespace legate::detail {

/**
 * @brief A cache that waits to free an allocation in hopes it is requested.
 *
 * To prevent spurious re-allocations, this cache wraps a user-supplied allocator and deleter
 * and upon attempts to return an allocation, if there is room in the cache, the allocation
 * will be saved so that later requests of the same size will return the cached allocation.
 *
 * There is room in the cache to store a returning allocation of size `size` if
 * - there are fewer than MAX_BIN_COUNT allocations of the same size that have been returned.
 * - the total size of allocations in the cache plus `size` does not exceed the maximum total size.
 *
 * If there is no room in the cache because it would exceed the maximum total size,
 * the contents of the cache are flushed and all allocations are freed, including the currently
 * returned allocation. If there is no room in the cache due to exceeding the same number of
 * allocations, only the currently returned allocation is flushed and freed..
 *
 * The user must provide allocator and deleter functions that the cache will use to allocate
 * and free memory.
 */
class AllocationCache {
 public:
  // The maximum number of allocations that can be cached of the same size.
  static constexpr std::size_t MAX_BIN_COUNT = 10;

  /**
   * @brief Signature for user-supplied allocation function.
   *
   * @param size The size of the allocation to allocate.
   *
   * @return A pointer to a newly created allocation of that size.
   */
  using Allocator = std::function<void*(std::uint64_t)>;
  /**
   * @brief Signature for user-supplied deletion function.
   *
   * @param ptr The pointer to the allocation to free.
   */
  using Deleter = std::function<void(void*)>;

  /**
   * @brief Constructor for the allocation cache.
   *
   * @param allocator The function to use to allocate pointers of a size.
   * @param deleter The function to use to free pointers given their address.
   * @param max_size The maximum total free bytes that can be held by the cache without freeing.
   *
   * @return A new, empty allocation cache.
   */
  explicit AllocationCache(Allocator allocator, Deleter deleter, std::uint64_t max_size);

  /**
   * @brief Destructor for the allocation cache.
   *
   * Flushes all allocations held within the cache via deleter_ prior to destruction.
   */
  ~AllocationCache();

  /**
   * @brief Get an allocation from the cache.
   *
   * The allocation policy is as follows: if there are no cached allocations of
   * the same size, defer to allocate_ to create a new allocation. Otherwise,
   * return the next cached allocation of the same size.
   *
   * @param size The size of the allocation to get.
   *
   * @return A pointer to the allocation.
   */
  [[nodiscard]] void* get_allocation(std::uint64_t size);

  /**
   * @brief Return an allocation to the cache.
   *
   * The return policy is as follows: if the total size of the cached
   * allocations exceeds the max total size, free the allocation and
   * flush the cache and free all allocations.
   *
   * Otherwise, if there is room to cache the allocation
   * (i.e. less than MAX_BIN_COUNT allocations of the same size), add it to
   * to the cache.
   *
   * Otherwise, free the allocation.
   *
   * @param ptr The pointer to the allocation to return.
   * @param size The size of the allocation to return.
   */
  void return_allocation(const void* ptr, std::uint64_t size);

 private:
  /**
   * @brief Flush all allocations held within the cache via deleter_.
   */
  void flush_free_allocations_();

  // The user supplied allocator and deleter functions for allocating and deleting memory.
  Allocator allocator_{};
  Deleter deleter_{};

  // A stack of allocations of the same size that are held by the cache and are not freed yet.
  using SizedAllocationStack = std::stack<void*, ::cuda::std::inplace_vector<void*, MAX_BIN_COUNT>>;

  // The map of allocations by size that are not yet freed but can be reused.
  std::unordered_map<std::uint64_t, SizedAllocationStack> allocations_;

  // The total size of all allocations held within the cache.
  std::uint64_t total_size_{0};

  // The maximum total size of all allocations that can be held within the cache.
  std::uint64_t max_total_size_{};
};

/**
 * @brief A cache for inline CPU storage.
 *
 * This cache is used for allocations made on the CPU with an optional alignment.
 */
class CpuAllocationCache : public AllocationCache {
 public:
  /**
   * @brief Constructor for the CPU allocation cache.
   *
   * This CpuAllocationCache utilizes new and delete to allocate and free bytes of memory.
   *
   * @param max_size The maximum total free bytes that can be held by the cache without freeing.
   * @param alignment The optional alignment to require for allocations. If std::nullopt, no
   * alignment is guaranteed.
   *
   * @return A new, empty CPU allocation cache.
   */
  explicit CpuAllocationCache(std::uint64_t max_size,
                              std::optional<std::uint8_t> alignment = std::nullopt) noexcept;
};

/**
 * @brief Attempt to advise the OS to use huge pages for the given allocation.
 *
 * Only done if the allocation is sufficiently large enough and if on Linux system.
 *
 * @param ptr The pointer to the allocation to advise.
 * @param size The size of the allocation to advise.
 */
void maybe_advise_huge_pages([[maybe_unused]] const void* ptr, [[maybe_unused]] std::size_t size);

}  // namespace legate::detail

#include <legate/data/detail/allocation_cache.inl>
