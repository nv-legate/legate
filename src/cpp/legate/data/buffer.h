/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/machine.h>
#include <legate/utilities/typedefs.h>

#include <legion.h>

#include <cstddef>
#include <cstdint>
#include <utility>

/**
 * @file
 * @brief Type alias definition for legate::Buffer and utility functions for it
 */

namespace legate {

namespace detail {

void check_alignment(std::size_t alignment);

}  // namespace detail

/**
 * @addtogroup data
 * @{
 */

/**
 * @brief The default alignment for memory allocations
 */
inline constexpr std::size_t DEFAULT_ALIGNMENT = 16;

/**
 * @brief A typed buffer class for intra-task temporary allocations
 *
 * Values in a buffer can be accessed by index expressions with \ref Point objects, or via a raw
 * pointer to the underlying allocation, which can be queried with the `Buffer::ptr()` method.
 *
 * \ref Buffer is an alias to
 * [Legion::DeferredBuffer](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion.h#L3509-L3609).
 *
 * Note on using temporary buffers in CUDA tasks:
 *
 * We use `Legion::DeferredBuffer`, whose lifetime is not connected with the CUDA stream(s)
 * used to launch kernels. The buffer is allocated immediately at the point when
 * create_buffer() is called, whereas the kernel that uses it is placed on a stream, and may
 * run at a later point. Normally a `Legion::DeferredBuffer` is deallocated automatically by
 * Legion once all the kernels launched in the task are complete. However, a
 * `Legion::DeferredBuffer` can also be deallocated immediately using
 * `Legion::DeferredBuffer::destroy()`, which is useful for operations that want to deallocate
 * intermediate memory as soon as possible. This deallocation is not synchronized with the task
 * stream, i.e. it may happen before a kernel which uses the buffer has actually
 * completed. This is safe as long as we use the same stream on all GPU tasks running on the
 * same device (which is guaranteed by the current implementation of @ref
 * TaskContext::get_task_stream()), because then all the actual uses of the buffer are done in
 * order on the one stream. It is important that all library CUDA code uses @ref
 * TaskContext::get_task_stream(), and all CUDA operations (including library calls) are
 * enqueued on that stream exclusively. This analysis additionally assumes that no code outside
 * of Legate is concurrently allocating from the eager pool, and that it's OK for kernels to
 * access a buffer even after it's technically been deallocated.
 */
template <typename VAL, std::int32_t DIM = 1>
using Buffer = Legion::DeferredBuffer<VAL, DIM>;

/**
 * @brief Creates a \ref Buffer of specific extents
 *
 * @param extents Extents of the buffer
 * @param kind Kind of the target memory (optional). If not given, the runtime will pick
 * automatically based on the executing processor
 * @param alignment Alignment for the memory allocation (optional)
 *
 * @return A \ref Buffer object
 */
template <typename VAL, std::int32_t DIM>
[[nodiscard]] Buffer<VAL, DIM> create_buffer(const Point<DIM>& extents,
                                             Memory::Kind kind     = Memory::Kind::NO_MEMKIND,
                                             std::size_t alignment = DEFAULT_ALIGNMENT)
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  detail::check_alignment(alignment);

  if (Memory::Kind::NO_MEMKIND == kind) {
    kind = find_memory_kind_for_executing_processor(false);
  }
  auto hi = extents - Point<DIM>::ONES();
  return Buffer<VAL, DIM>{Rect<DIM>{Point<DIM>::ZEROES(), std::move(hi)}, kind, nullptr, alignment};
}

/**
 * @brief Creates a \ref Buffer of a specific size. Always returns a 1D \ref Buffer.
 *
 * @param size Size of the \ref Buffer
 * @param kind `Memory::Kind` of the target memory (optional). If not given, the runtime will
 * pick automatically based on the executing processor
 * @param alignment Alignment for the memory allocation (optional)
 *
 * @return A 1D \ref Buffer object
 */
template <typename VAL>
[[nodiscard]] Buffer<VAL> create_buffer(std::size_t size,
                                        Memory::Kind kind     = Memory::Kind::NO_MEMKIND,
                                        std::size_t alignment = DEFAULT_ALIGNMENT)
{
  return create_buffer<VAL, 1>(Point<1>{size}, kind, alignment);
}

/** @} */

}  // namespace legate
