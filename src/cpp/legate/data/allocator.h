/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <memory>

/**
 * @file
 * @brief Class definition for legate::ScopedAllocator
 */

namespace legate {

/**
 * @addtogroup data
 * @{
 */

/**
 * @brief A simple allocator backed by \ref Buffer objects
 *
 * For each allocation request, this allocator creates a 1D \ref Buffer of
 * `std::int8_t` and returns the raw pointer to it. By default, all allocations
 * are deallocated when the allocator is destroyed, and can optionally be made
 * alive until the task finishes by making the allocator unscoped.
 *
 * `ScopedAllocator` is copyable (primarily so types derived from
 * `ScopedAllocator` can satisfy the `Allocator named requirement
 * <https://en.cppreference.com/w/cpp/named_req/Allocator.html>`_), but all copies share a reference
 * to the same `ScopedAllocator::Impl`, so scoped deallocation will not occur until all copies go
 * out of scope.
 */
class LEGATE_EXPORT ScopedAllocator {
 public:
  static constexpr std::size_t DEFAULT_ALIGNMENT = 16;

  /**
   * @brief Create a `ScopedAllocator` for a specific memory kind
   *
   * @param kind `Memory::Kind` of the memory on which the \ref Buffer should be created
   * @param scoped If true, the allocator is scoped; i.e., lifetimes of allocations are tied to
   * the allocator's lifetime. Otherwise, the allocations are alive until the task finishes
   * (and unless explicitly deallocated).
   * @param alignment Alignment for the allocations from `allocate()` (use
   * `allocate_aligned()` to specify a different alignment)
   *
   * @throws std::domain_error If `alignment` is 0, or not a power of 2.
   */
  explicit ScopedAllocator(Memory::Kind kind,
                           bool scoped           = true,
                           std::size_t alignment = DEFAULT_ALIGNMENT);

  ~ScopedAllocator() noexcept;

  ScopedAllocator(const ScopedAllocator&)            = default;
  ScopedAllocator& operator=(const ScopedAllocator&) = default;

  /**
   * @brief Allocates a contiguous buffer of the given `Memory::Kind`
   *
   * When the allocator runs out of memory, the runtime will fail with an error message.
   * Otherwise, the function returns a valid pointer. If `bytes` is `0`, returns `nullptr`.
   *
   * @param bytes Size of the allocation in bytes
   *
   * @return A raw pointer to the allocation
   *
   * @see deallocate
   * @see allocate_aligned
   */
  [[nodiscard]] void* allocate(std::size_t bytes);

  /**
   * @brief Allocates a contiguous buffer of the given `Memory::Kind` with a
   * specified alignment.
   *
   * @param bytes Size of the allocation in bytes.
   * @param alignment Alignment in bytes of this allocation.
   *
   * @return A raw pointer to the allocation
   *
   * @see deallocate
   *
   * @throws std::domain_error If `alignment` is 0, or not a power of 2.
   *
   * @see allocate_aligned
   */
  [[nodiscard]] void* allocate_aligned(std::size_t bytes, std::size_t alignment);

  /**
   * @brief Allocates a contiguous buffer of uninitialized `T`s with the given
   * `Memory::Kind`.
   *
   * @param num_items The number of items to allocate
   *
   * @return A raw pointer to the allocation
   *
   * @see deallocate
   */
  template <typename T>
  [[nodiscard]] T* allocate_type(std::size_t num_items);

  /**
   * @brief Deallocates an allocation.
   *
   * @param ptr Pointer to the allocation to deallocate
   *
   * @throws std::invalid_argument If `ptr` was not allocated by this allocator.
   *
   * The input pointer must be one that was previously returned by an `allocate()` call. If
   * `ptr` is `nullptr`, this call does nothing.
   *
   * @see allocate
   */
  void deallocate(void* ptr);

 private:
  class Impl;

  std::shared_ptr<Impl> impl_{};
};

/** @} */

}  // namespace legate

#include <legate/data/allocator.inl>
