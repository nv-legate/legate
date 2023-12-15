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

#include "core/utilities/machine.h"
#include "core/utilities/typedefs.h"

#include "legion.h"

#include <cstddef>

/**
 * @file
 * @brief Type alias definition for legate::Buffer and utility functions for it
 */

namespace legate {

inline constexpr size_t DEFAULT_ALIGNMENT = 16;

/**
 * @ingroup data
 * @brief A typed buffer class for intra-task temporary allocations
 *
 * Values in a buffer can be accessed by index expressions with legate::Point objects,
 * or via a raw pointer to the underlying allocation, which can be queried with the `ptr` method.
 *
 * `legate::Buffer` is an alias to
 * [`Legion::DeferredBuffer`](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion.h#L3509-L3609).
 *
 * Note on using temporary buffers in CUDA tasks:
 *
 * We use Legion `DeferredBuffer`, whose lifetime is not connected with the CUDA stream(s) used to
 * launch kernels. The buffer is allocated immediately at the point when `create_buffer` is called,
 * whereas the kernel that uses it is placed on a stream, and may run at a later point. Normally
 * a `DeferredBuffer` is deallocated automatically by Legion once all the kernels launched in the
 * task are complete. However, a `DeferredBuffer` can also be deallocated immediately using
 * `destroy()`, which is useful for operations that want to deallocate intermediate memory as soon
 * as possible. This deallocation is not synchronized with the task stream, i.e. it may happen
 * before a kernel which uses the buffer has actually completed. This is safe as long as we use the
 * same stream on all GPU tasks running on the same device (which is guaranteed by the current
 * implementation of `get_cached_stream`), because then all the actual uses of the buffer are done
 * in order on the one stream. It is important that all library CUDA code uses
 * `get_cached_stream()`, and all CUDA operations (including library calls) are enqueued on that
 * stream exclusively. This analysis additionally assumes that no code outside of Legate is
 * concurrently allocating from the eager pool, and that it's OK for kernels to access a buffer even
 * after it's technically been deallocated.
 */
template <typename VAL, int32_t DIM = 1>
using Buffer = Legion::DeferredBuffer<VAL, DIM>;

/**
 * @ingroup data
 * @brief Creates a `Buffer` of specific extents
 *
 * @param extents Extents of the buffer
 * @param kind Kind of the target memory (optional). If not given, the runtime will pick
 * automatically based on the executing processor
 * @param alignment Alignment for the memory allocation (optional)
 *
 * @return A `Buffer` object
 */
template <typename VAL, int32_t DIM>
Buffer<VAL, DIM> create_buffer(const Point<DIM>& extents,
                               Memory::Kind kind = Memory::Kind::NO_MEMKIND,
                               size_t alignment  = DEFAULT_ALIGNMENT)
{
  static_assert(DIM <= LEGATE_MAX_DIM);

  if (Memory::Kind::NO_MEMKIND == kind) {
    kind = find_memory_kind_for_executing_processor(false);
  }
  auto hi = extents - Point<DIM>::ONES();
  // We just avoid creating empty buffers, as they cause all sorts of headaches.
  for (int32_t idx = 0; idx < DIM; ++idx) {
    hi[idx] = std::max<int64_t>(hi[idx], 0);
  }
  return Buffer<VAL, DIM>{Rect<DIM>{Point<DIM>::ZEROES(), std::move(hi)}, kind, nullptr, alignment};
}

/**
 * @ingroup data
 * @brief Creates a `Buffer` of a specific size. Always returns a 1D buffer.
 *
 * @param size Size of the buffdr
 * @param kind Kind of the target memory (optional). If not given, the runtime will pick
 * automatically based on the executing processor
 * @param alignment Alignment for the memory allocation (optional)
 *
 * @return A 1D `Buffer` object
 */
template <typename VAL>
Buffer<VAL> create_buffer(size_t size,
                          Memory::Kind kind = Memory::Kind::NO_MEMKIND,
                          size_t alignment  = DEFAULT_ALIGNMENT)
{
  return create_buffer<VAL, 1>(Point<1>(size), kind, alignment);
}

}  // namespace legate
