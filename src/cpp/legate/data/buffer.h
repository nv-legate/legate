/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/data/raw_allocation.h>
#include <legate/geometry/point.h>
#include <legate/geometry/rect.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <legion.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

/**
 * @file
 * @brief Class definitions for legate::Buffer and legate::NDBuffer and utility functions for them
 */

namespace legate::mapping {

enum class StoreTarget : std::uint8_t;

}  // namespace legate::mapping

namespace legate {

class PhysicalStore;

}  // namespace legate

namespace legate::detail {

class Buffer;

}  // namespace legate::detail

namespace legate {

/**
 * @addtogroup data
 * @{
 */

/**
 * @brief The default alignment for memory allocations
 */
inline constexpr std::size_t DEFAULT_BUFFER_ALIGNMENT = 16;

/**
 * @brief A type-erased buffer
 *
 * A `Buffer` is, as the name implies, "local" to a task. Its lifetime is bound to that of the
 * task. When the task ends, the buffer is destroyed. It is most commonly used as temporary
 * scratch-space within tasks for that reason.
 *
 * The buffer is allocated immediately at the point when `Buffer` is created, so it is safe to use
 * it immediately, even if it used asynchronously (for example, in GPU kernel launches) after the
 * fact.
 *
 * A `Buffer` is type-erased: it knows its element `Type`, dimension, and domain, but not a static
 * element type. Direct access to the underlying data (and the ability to create accessors) is only
 * possible through a typed \ref NDBuffer, which can be obtained by wrapping a `Buffer`.
 */
class LEGATE_EXPORT Buffer {
 public:
  /**
   * @brief Construct a `Buffer` viewing an existing raw allocation.
   *
   * @param allocation The raw allocation backing the buffer.
   * @param type The type to interpret `allocation` as.
   * @param domain The extents of the buffer.
   */
  Buffer(const RawAllocation& allocation, const Type& type, Rect domain);
  /**
   * @return The type of the buffer.
   */
  [[nodiscard]] Type type() const;

  /**
   * @return The shape of the buffer.
   */
  [[nodiscard]] const Rect& domain() const;

  /**
   * @return The dimension of the buffer
   */
  [[nodiscard]] std::uint32_t dim() const;

  /**
   * @return The memory kind of the buffer.
   */
  [[nodiscard]] mapping::StoreTarget target() const;

  /**
   * @brief Returns the base pointer and per-dimension element strides of the buffer's affine
   * layout.
   *
   * The base pointer is the address of the buffer's coordinate origin (the point `(0, ..., 0)`),
   * following the affine convention `addr(p) = base + type().size() * dot(p, strides)`. The strides
   * are in element units (not bytes). This is the layout information used to build a numpy/DLPack
   * view of the buffer.
   *
   * @return A pair of the base pointer and the element strides.
   */
  [[nodiscard]] std::pair<void*, Point> base_and_element_strides() const;

  [[nodiscard]] const void* ptr(const Point& point) const;

  /**
   * @return The raw allocation backing the buffer (used to bind the buffer to a store or to wrap
   * it as a typed \ref NDBuffer).
   */
  [[nodiscard]] RawAllocation allocation() const;

  [[nodiscard]] const SharedPtr<detail::Buffer>& impl() const;

  void reshape(Rect new_domain);

  /**
   * @brief Construct a freshly allocated `Buffer`.
   *
   * @param type The type of the buffer.
   * @param domain The extents of the buffer.
   * @param target The kind of memory to allocate.
   */
  [[nodiscard]] static Buffer allocate(const Type& type, Rect domain, mapping::StoreTarget target);

  Buffer() = default;
  Buffer(const Buffer&);
  Buffer& operator=(const Buffer&);
  Buffer(Buffer&&) noexcept;
  Buffer& operator=(Buffer&&) noexcept;
  ~Buffer();

  explicit Buffer(InternalSharedPtr<detail::Buffer> impl);

 private:
  SharedPtr<detail::Buffer> impl_{};
};

/**
 * @brief A typed view into a \ref Buffer.
 *
 * Values in an `NDBuffer` can be accessed by index expressions with \ref NDPoint objects, or via a
 * raw pointer to the underlying allocation, which can be queried with the `NDBuffer::ptr()` method.
 *
 * An `NDBuffer` is a lightweight typed wrapper around a \ref Buffer. It shares ownership of the
 * `Buffer`'s backing storage, so the memory stays alive for as long as any copy of the buffer (or
 * the store the buffer is bound to) is alive. The allocation may also be released eagerly with
 * `NDBuffer::destroy()`.
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
class NDBuffer {
 public:
  /**
   * @brief Returns a reference to the element at `p`.
   *
   * @param p The point to access (in absolute coordinates within `get_bounds()`).
   *
   * @return A reference to the element.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE VAL& operator[](const NDPoint<DIM>& p) const;

  /**
   * @brief Returns a pointer to the element at `p`.
   *
   * @param p The point to access.
   *
   * @return A pointer to the element.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE VAL* ptr(const NDPoint<DIM>& p) const;

  /**
   * @brief Returns a pointer to the element at `r.lo`, additionally writing the element strides.
   *
   * @param r The rectangle whose lower bound to return a pointer to.
   * @param strides Out-param that receives the per-dimension element strides.
   *
   * @return A pointer to the element at `r.lo`.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE VAL* ptr(const NDRect<DIM>& r, std::size_t strides[DIM]) const;

  /**
   * @return The extents of the buffer.
   */
  [[nodiscard]] LEGATE_HOST_DEVICE NDRect<DIM> domain() const;

  /**
   * @return The type-erased \ref Buffer this view wraps (used to bind the buffer to a store).
   */
  [[nodiscard]] const Buffer& buffer() const;

  /**
   * @brief Eagerly releases this buffer's ownership of the backing storage.
   *
   * If this buffer holds the last reference to the storage, the allocation is freed.
   */
  void destroy();

  NDBuffer() = default;

  /**
   * @brief Wraps a type-erased \ref Buffer as a typed `NDBuffer`.
   *
   * @param buffer The buffer to view.
   *
   * @throws std::invalid_argument If `sizeof(VAL)` is not the same as the buffer's type size.
   * @throws std::invalid_argument If `alignof(VAL)` is not the same as the buffer's type alignment.
   */
  explicit NDBuffer(Buffer buffer);

 private:
  struct InitTag {};

  // Wraps `buffer` without validating that its type matches `VAL`. Used by the runtime factory
  // paths (create_buffer / PhysicalStore::create_output_buffer) where the type is known to match
  // by construction.
  NDBuffer(Buffer buffer, InitTag);

  // Fills the cached base pointer, strides, and bounds from `buffer_`.
  void init_();

  template <typename V, std::int32_t D>
  friend NDBuffer<V, D> create_buffer(const NDPoint<D>&,
                                      std::optional<mapping::StoreTarget>,
                                      std::size_t);
  friend class PhysicalStore;

  Buffer buffer_{};
  VAL* base_{nullptr};      // address of coordinate (0, ..., 0)
  NDPoint<DIM> strides_{};  // element strides
  NDRect<DIM> domain_{};
};

/**
 * @brief Creates an \ref NDBuffer of specific extents
 *
 * @param extents Extents of the buffer
 * @param target Target of the buffer's memory (optional). If not given, the runtime will pick
 * automatically based on the executing processor
 * @param alignment Alignment for the memory allocation (optional)
 *
 * @return An \ref NDBuffer object
 */
template <typename VAL, std::int32_t DIM>
[[nodiscard]] NDBuffer<VAL, DIM> create_buffer(
  const NDPoint<DIM>& extents,
  std::optional<mapping::StoreTarget> target = std::nullopt,
  std::size_t alignment                      = DEFAULT_BUFFER_ALIGNMENT);

/**
 * @brief Creates an \ref NDBuffer of a specific size. Always returns a 1D \ref NDBuffer.
 *
 * @param size Size of the \ref NDBuffer
 * @param target Target of the buffer's memory (optional). If not given, the runtime will
 * pick automatically based on the executing processor
 * @param alignment Alignment for the memory allocation (optional)
 *
 * @return A 1D \ref NDBuffer object
 */
template <typename VAL>
[[nodiscard]] NDBuffer<VAL> create_buffer(std::size_t size,
                                          std::optional<mapping::StoreTarget> target = std::nullopt,
                                          std::size_t alignment = DEFAULT_BUFFER_ALIGNMENT);

/** @} */

}  // namespace legate

#include <legate/data/buffer.inl>
