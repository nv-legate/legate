/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "core/utilities/typedefs.h"

#include <cstddef>
#include <memory>

/**
 * @file
 * @brief Class definition for legate::ScopedAllocator
 */

namespace legate {

/**
 * @ingroup data
 * @brief A simple allocator backed by `Buffer` objects
 *
 * For each allocation request, this allocator creates a 1D `Buffer` of `int8_t` and returns
 * the raw pointer to it. By default, all allocations are deallocated when the allocator is
 * destroyed, and can optionally be made alive until the task finishes by making the allocator
 * unscoped.
 */
class ScopedAllocator {
 public:
  static inline constexpr size_t DEFAULT_ALIGNMENT = 16;

  // Iff 'scoped', all allocations will be released upon destruction.
  // Otherwise this is up to the runtime after the task has finished.
  /**
   * @brief Create a `ScopedAllocator` for a specific memory kind
   *
   * @param kind Kind of the memory on which the `Buffer`s should be created
   * @param scoped If true, the allocator is scoped; i.e., lifetimes of allocations are tied to
   * the allocator's lifetime. Otherwise, the allocations are alive until the task finishes
   * (and unless explicitly deallocated).
   * @param alignment Alignment for the allocations
   */
  ScopedAllocator(Memory::Kind kind, bool scoped = true, size_t alignment = DEFAULT_ALIGNMENT);

  /**
   * @brief Allocates a contiguous buffer of the given Memory::Kind
   *
   * When the allocator runs out of memory, the runtime will fail with an error message.
   * Otherwise, the function returns a valid pointer.
   *
   * @param bytes Size of the allocation in bytes
   *
   * @return A raw pointer to the allocation
   */
  [[nodiscard]] void* allocate(size_t bytes);
  /**
   * @brief Deallocates an allocation. The input pointer must be one that was previously
   * returned by an `allocate` call, otherwise the code will fail with an error message.
   *
   * @param ptr Pointer to the allocation to deallocate
   */
  void deallocate(void* ptr);

 private:
  class Impl;

  // See StoreMapping::StoreMappingImplDeleter for why this this exists
  struct ImplDeleter {
    void operator()(Impl* ptr) const noexcept;
  };

  std::unique_ptr<Impl, ImplDeleter> impl_{};
};

}  // namespace legate
