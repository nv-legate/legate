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

#pragma once

#include "legate/utilities/typedefs.h"

#include <cstddef>
#include <memory>

/**
 * @file
 * @brief Class definition for legate::ScopedAllocator
 */

namespace legate {

/**
 * @ingroup data
 * @brief A simple allocator backed by \ref Buffer objects
 *
 * For each allocation request, this allocator creates a 1D \ref Buffer of `std::int8_t` and
 * returns the raw pointer to it. By default, all allocations are deallocated when the allocator is
 * destroyed, and can optionally be made alive until the task finishes by making the allocator
 * unscoped.
 */
class ScopedAllocator {
 public:
  static constexpr std::size_t DEFAULT_ALIGNMENT = 16;

  /**
   * @brief Create a `ScopedAllocator` for a specific memory kind
   *
   * @param kind `Memory::Kind` of the memory on which the \ref Buffer should be created
   * @param scoped If true, the allocator is scoped; i.e., lifetimes of allocations are tied to
   * the allocator's lifetime. Otherwise, the allocations are alive until the task finishes
   * (and unless explicitly deallocated).
   * @param alignment Alignment for the allocations
   *
   * @throws std::domain_error If `alignment` is 0, or not a power of 2.
   */
  explicit ScopedAllocator(Memory::Kind kind,
                           bool scoped           = true,
                           std::size_t alignment = DEFAULT_ALIGNMENT);

  ~ScopedAllocator() noexcept;

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
   */
  [[nodiscard]] void* allocate(std::size_t bytes);

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

  std::unique_ptr<Impl> impl_{};
};

}  // namespace legate
