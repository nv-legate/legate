/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/raw_allocation.h>
#include <legate/geometry/point.h>
#include <legate/geometry/rect.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <utility>

namespace legate::detail {

/**
 * @brief A memory allocation with a shape and a type
 */
class Buffer {
 public:
  /**
   * @brief Construct a `Buffer`.
   *
   * @param allocation The raw allocation backing this buffer.
   * @param type The type to interpret `allocation` as.
   * @param domain The extents of the buffer.
   * @param dim_order The dimensions of `domain`, listed from the most rapidly changing one to the
   * least rapidly changing one (i.e. `dim_order[0]` is the unit-stride dimension). An empty
   * `dim_order` means C ordering. See `mapping::detail::DimOrdering::dimension_order()`.
   */
  Buffer(InternalSharedPtr<RawAllocation> allocation,
         InternalSharedPtr<Type> type,
         Rect domain,
         SmallVector<std::int32_t, LEGATE_MAX_DIM> dim_order = {});

  /**
   * @return The type of the buffer.
   */
  [[nodiscard]] const InternalSharedPtr<Type>& type() const;

  /**
   * @return The dimension of the buffer
   */
  [[nodiscard]] std::uint32_t dim() const;

  /**
   * @return The shape of the buffer.
   */
  [[nodiscard]] const Rect& domain() const;

  /**
   * @return The memory kind of the buffer.
   */
  [[nodiscard]] mapping::StoreTarget target() const;

  /**
   * @return The raw allocation backing this buffer.
   */
  [[nodiscard]] const InternalSharedPtr<RawAllocation>& allocation() const;

  /**
   * @brief Returns a raw pointer to the memory allocation managed by the buffer.
   */
  [[nodiscard]] void* data();

  /**
   * @brief Returns a const raw pointer to the memory allocation managed by the buffer.
   */
  [[nodiscard]] const void* data() const;

  /**
   * @brief Returns the base pointer and per-dimension element strides of the buffer's affine
   * layout.
   *
   * The base pointer is the address of the element at the buffer's coordinate origin (i.e. the
   * point `(0, ..., 0)`), following the affine convention `addr(p) = base + field_size * sum(p[i]
   * * strides[i])`. The strides are given in elements (not bytes) and have as many entries as the
   * buffer domain's dimension (at least one).
   *
   * @return A pair of the base pointer and the element strides.
   */
  [[nodiscard]] std::pair<void*, Point> base_and_element_strides() const;

  /**
   * @brief Returns the per-dimension element strides of the buffer's dense layout.
   *
   * The strides follow the buffer's dimension ordering, so they are not necessarily C-order
   * strides of the domain.
   *
   * @return The element strides.
   */
  [[nodiscard]] Point element_strides() const;

  /**
   * @return The buffer's dimension ordering, from the most rapidly changing dimension to the
   * least rapidly changing one. Empty for a 0-D buffer.
   */
  [[nodiscard]] Span<const std::int32_t> dim_order() const;

  void reshape(Rect new_domain);

  /**
   * @brief Creates a `Buffer` by allocating fresh memory on the given target.
   *
   * Unlike the constructors, which view an already-existing `RawAllocation`, this factory allocates
   * the backing memory itself. The lifetime of the allocated memory is tied to the lifetime of the
   * returned buffer (and any other holder of the underlying allocation).
   *
   * Freshly allocated buffers always have C ordering.
   *
   * @param type The type to interpret the allocation as.
   * @param domain The extents of the buffer.
   * @param target The target memory kind to allocate on.
   *
   * @return The freshly allocated buffer.
   */
  [[nodiscard]] static InternalSharedPtr<Buffer> allocate(InternalSharedPtr<Type> type,
                                                          Rect domain,
                                                          mapping::StoreTarget target);

 private:
  InternalSharedPtr<RawAllocation> allocation_{};
  InternalSharedPtr<Type> type_{};
  Rect domain_{};
  // Dimensions from the most rapidly changing to the least rapidly changing. Always has `dim()`
  // entries once constructed; the constructor fills in C ordering when none is given.
  SmallVector<std::int32_t, LEGATE_MAX_DIM> dim_order_{};
};

}  // namespace legate::detail

#include <legate/data/detail/buffer.inl>
