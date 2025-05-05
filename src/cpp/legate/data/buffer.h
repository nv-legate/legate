/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/inline_allocation.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <legion.h>

#include <cstddef>
#include <cstdint>
#include <optional>

/**
 * @file
 * @brief Type alias definition for legate::Buffer and utility functions for it
 */

namespace legate::mapping {

enum class StoreTarget : std::uint8_t;

}  // namespace legate::mapping

namespace legate::detail {

class TaskLocalBuffer;

}  // namespace legate::detail

namespace legate {

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
 * @brief A task-local temporary buffer.
 *
 * A `TaskLocalBuffer` is, as the name implies, "local" to a task. Its lifetime is bound to
 * that of the task. When the task ends, the buffer is destroyed. It is most commonly used as
 * temporary scratch-space within tasks for that reason.
 *
 * The buffer is allocated immediately at the point when `TaskLocalBuffer` is created, so it is
 * safe to use it immediately, even if it used asynchronously (for example, in GPU kernel
 * launches) after the fact.
 */
class TaskLocalBuffer {
 public:
  TaskLocalBuffer() = LEGATE_DEFAULT_WHEN_CYTHON;
  TaskLocalBuffer(const TaskLocalBuffer&);
  TaskLocalBuffer& operator=(const TaskLocalBuffer&);
  TaskLocalBuffer(TaskLocalBuffer&&) noexcept;
  TaskLocalBuffer& operator=(TaskLocalBuffer&&) noexcept;
  ~TaskLocalBuffer();

  explicit TaskLocalBuffer(SharedPtr<detail::TaskLocalBuffer> impl);

  /**
   * @brief Construct a `TaskLocalBuffer`.
   *
   * @param buf The Legion buffer from which to construct this buffer.
   * @param type The type to interpret `buf` as.
   * @param bounds The extent of the buffer.
   */
  TaskLocalBuffer(const Legion::UntypedDeferredBuffer<>& buf,
                  const Type& type,
                  const Domain& bounds);

  /**
   * @brief Construct a `TaskLocalBuffer`.
   *
   * If `mem_kind` is not given, the memory kind is automatically deduced based on the type of
   * processor executing the task. For example, GPU tasks will allocate GPU memory (pure GPU
   * memory that is, not zero-copy), while CPU tasks will allocate regular system memory.
   *
   * @param type The type of the buffer.
   * @param bounds The extents of the buffer.
   * @param mem_kind The kind of memory to allocate.
   */
  TaskLocalBuffer(const Type& type,
                  Span<const std::uint64_t> bounds,
                  std::optional<mapping::StoreTarget> mem_kind = std::nullopt);

  /**
   * @brief Construct a `TaskLocalBuffer`.
   *
   * If this kind of constructor is used, the user should almost always prefer the `type`-less
   * version of this ctor. That constructor will deduce the `Type` based on `T`. The point of
   * _this_ ctor is to provide the ability to type-pun the `Buffer` with an equivalent type.
   *
   * @param buf The typed Legion buffer from which to construct this buffer from.
   * @param type The type to interpret `buf` as.
   *
   * @throws std::invalid_argument If `sizeof(T)` is not the same as the type size.
   * @throws std::invalid_argument If `alignof(T)` is not the same as the type alignment.
   */
  template <typename T, std::int32_t DIM>
  TaskLocalBuffer(const Buffer<T, DIM>& buf, const Type& type);

  /**
   * @brief Construct a `TaskLocalBuffer`.
   *
   * The type of the buffer is deduced from `T`.
   *
   * @param buf The typed Legion buffer from which to construct this buffer from.
   */
  template <typename T, std::int32_t DIM>
  explicit TaskLocalBuffer(const Buffer<T, DIM>& buf);

  /**
   * @return The type of the buffer.
   */
  [[nodiscard]] Type type() const;

  /**
   * @return The dimension of the buffer
   */
  [[nodiscard]] std::int32_t dim() const;

  /**
   * @return The shape of the buffer.
   */
  [[nodiscard]] const Domain& domain() const;

  /**
   * @return The memory kind of the buffer.
   */
  [[nodiscard]] mapping::StoreTarget memory_kind() const;

  /**
   * @brief Convert this object to a typed `Buffer`.
   *
   * Since `TaskLocalBuffer` is type-erased, there is not a whole lot you can do with it
   * normally. Access to the underlying data (and ability to create accessors) is only possible
   * with a typed buffer.
   *
   * @return The typed buffer.
   */
  template <typename T, std::int32_t DIM>
  [[nodiscard]] explicit operator Buffer<T, DIM>() const;

  /**
   * @brief Get the `InlineAllocation` for the buffer.
   *
   * This routine constructs a fresh `InlineAllocation` for each call. This process may not be
   * cheap, so the user is encouraged to call this sparingly.
   *
   * @return The inline allocation object.
   */
  [[nodiscard]] InlineAllocation get_inline_allocation() const;

  [[nodiscard]] const SharedPtr<detail::TaskLocalBuffer>& impl() const;

 private:
  [[nodiscard]] const Legion::UntypedDeferredBuffer<>& legion_buffer_() const;

  SharedPtr<detail::TaskLocalBuffer> impl_{};
};

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
                                             std::size_t alignment = DEFAULT_ALIGNMENT);

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
                                        std::size_t alignment = DEFAULT_ALIGNMENT);

/** @} */

}  // namespace legate

#include <legate/data/buffer.inl>
