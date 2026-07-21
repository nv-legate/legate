/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/data/inline_allocation.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <legion.h>

#include <cstddef>
#include <cstdint>
#include <optional>

/**
 * @file
 * @brief Class definition for legate::Buffer and utility functions for it
 */

namespace legate::mapping {

enum class StoreTarget : std::uint8_t;

}  // namespace legate::mapping

namespace legate::detail {

class InlineStorage;
class TaskLocalBuffer;

/**
 * @brief Allocates an `InlineStorage` of `field_size`-byte elements spanning `bounds`.
 *
 * Exported so that the `Buffer` class template (which is instantiated in user code) never has to
 * touch `InlineStorage`'s (hidden-visibility) members directly.
 *
 * @param field_size The size of each element in bytes.
 * @param bounds The domain the storage should span.
 * @param kind The memory kind to allocate on.
 *
 * @return The freshly allocated inline storage.
 */
[[nodiscard]] LEGATE_EXPORT InternalSharedPtr<InlineStorage> create_buffer_storage(
  std::uint32_t field_size, const Rect& bounds, Memory::Kind kind);

/**
 * @brief Reads the base pointer and element strides of an inline storage's affine layout.
 *
 * `base_out` receives the address of the storage's coordinate origin (i.e. the point
 * `(0, ..., 0)`); `element_strides_out` (which must have room for at least `LEGATE_MAX_DIM`
 * entries) receives the per-dimension element strides (not bytes).
 *
 * @param storage The storage to inspect.
 * @param base_out Out-param for the base pointer.
 * @param element_strides_out Out-param for the element strides.
 */
LEGATE_EXPORT void inline_storage_layout(const InternalSharedPtr<InlineStorage>& storage,
                                         void** base_out,
                                         std::uint64_t* element_strides_out);

[[nodiscard]] LEGATE_EXPORT Memory::Kind get_memory_kind(
  const InternalSharedPtr<InlineStorage>& storage);

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
 * Values in a buffer can be accessed by index expressions with \ref NDPoint objects, or via a raw
 * pointer to the underlying allocation, which can be queried with the `Buffer::ptr()` method.
 *
 * A \ref Buffer is backed by a `legate::detail::InlineStorage`, a manually-managed affine
 * allocation. The buffer shares ownership of that storage, so the memory stays alive for as long
 * as any copy of the buffer (or the store the buffer is bound to) is alive. The allocation may
 * also be released eagerly with `Buffer::destroy()`.
 *
 * The buffer uses a C-ordered (row-major) affine layout, consistent with the layout used by the
 * store accessors, so a buffer that is bound to an output store can be read back through the
 * store's accessors without any re-layout.
 *
 * @note The indexing operator and `ptr()` overloads only read the cached base pointer and strides,
 * so they are safe to call from device code. The buffer shares ownership through a smart pointer
 * whose operations must run on the host, so a buffer should be created (and destroyed) on the
 * host; the raw pointer obtained from `ptr()` is what should be handed to device kernels.
 */
template <typename VAL, std::int32_t DIM = 1>
class Buffer {
 public:
  Buffer() = default;

  /**
   * @brief Constructs a `Buffer` that views `storage` over `bounds`.
   *
   * @param storage The inline storage backing the buffer.
   * @param bounds The extents (as a rectangle whose lower bound is the origin) of the buffer.
   */
  Buffer(InternalSharedPtr<detail::InlineStorage> storage, const NDRect<DIM>& bounds);

  /**
   * @brief Returns a reference to the element at `p`.
   *
   * @param p The point to access (in absolute coordinates within `get_bounds()`).
   *
   * @return A reference to the element.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE VAL& operator[](const NDPoint<DIM>& p) const { return *ptr(p); }

  /**
   * @brief Returns a pointer to the element at `p`.
   *
   * @param p The point to access.
   *
   * @return A pointer to the element.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE VAL* ptr(const NDPoint<DIM>& p) const
  {
    return base_ + offset_(p);
  }

  /**
   * @brief Returns a pointer to the element at `r.lo`, additionally writing the element strides.
   *
   * @param r The rectangle whose lower bound to return a pointer to.
   * @param strides Out-param that receives the per-dimension element strides.
   *
   * @return A pointer to the element at `r.lo`.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE VAL* ptr(const NDRect<DIM>& r, std::size_t strides[DIM]) const
  {
    for (std::int32_t i = 0; i < DIM; ++i) {
      strides[i] = strides_[i];
    }
    return ptr(r.lo);
  }

  /**
   * @return The extents of the buffer.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE NDRect<DIM> get_bounds() const { return bounds_; }

  /**
   * @brief Eagerly releases this buffer's ownership of the backing storage.
   *
   * If this buffer holds the last reference to the storage, the allocation is freed.
   */
  void destroy()
  {
    storage_ = {};
    base_    = nullptr;
  }

  /**
   * @return The inline storage backing the buffer (used to bind the buffer to a store or to wrap
   * it as a `TaskLocalBuffer`).
   */
  [[nodiscard]] const InternalSharedPtr<detail::InlineStorage>& get_storage() const
  {
    return storage_;
  }

  [[nodiscard]] Memory::Kind memory_kind() const { return detail::get_memory_kind(get_storage()); }

 private:
  [[nodiscard]] LEGATE_HOST_DEVICE std::int64_t offset_(const NDPoint<DIM>& p) const
  {
    std::int64_t off = 0;

    for (std::int32_t i = 0; i < DIM; ++i) {
      off += static_cast<std::int64_t>(p[i]) * static_cast<std::int64_t>(strides_[i]);
    }
    return off;
  }

  InternalSharedPtr<detail::InlineStorage> storage_{};
  VAL* base_{nullptr};          // address of coordinate (0, ..., 0)
  std::size_t strides_[DIM]{};  // element strides
  NDRect<DIM> bounds_{};
};

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
class LEGATE_EXPORT TaskLocalBuffer {
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
   * @param storage The inline storage backing the buffer.
   * @param type The type to interpret `storage` as.
   * @param bounds The extent of the buffer.
   */
  TaskLocalBuffer(InternalSharedPtr<detail::InlineStorage> storage,
                  const Type& type,
                  const Rect& bounds);

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
   * @param buf The typed buffer from which to construct this buffer from.
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
   * @param buf The typed buffer from which to construct this buffer from.
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
  [[nodiscard]] const Rect& domain() const;

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
  [[nodiscard]] const InternalSharedPtr<detail::InlineStorage>& storage_() const;

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
[[nodiscard]] Buffer<VAL, DIM> create_buffer(const NDPoint<DIM>& extents,
                                             Memory::Kind kind     = Memory::Kind::NO_MEMKIND,
                                             std::size_t alignment = DEFAULT_ALIGNMENT);

/**
 * @brief Creates a \ref Buffer of specific extents.
 *
 * @param extents Extents of the buffer.
 * @param mem The target memory of the buffer.
 * @param alignment Alignment for the memory allocation (optional).
 *
 * @return A \ref Buffer object.
 */
template <typename VAL, std::int32_t DIM>
[[nodiscard]] Buffer<VAL, DIM> create_buffer(const NDPoint<DIM>& extents,
                                             Memory mem,
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

/**
 * @brief Creates a \ref Buffer of a specific size. Always returns a 1D \ref Buffer.
 *
 * @param size Size of the \ref Buffer.
 * @param mem The target memory of the buffer.
 * @param alignment Alignment for the memory allocation (optional).
 *
 * @return A 1D \ref Buffer object.
 */
template <typename VAL>
[[nodiscard]] Buffer<VAL> create_buffer(std::size_t size,
                                        Memory mem,
                                        std::size_t alignment = DEFAULT_ALIGNMENT);

/** @} */

}  // namespace legate

#include <legate/data/buffer.inl>
