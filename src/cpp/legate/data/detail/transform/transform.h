/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/restriction.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <iosfwd>

namespace legate::detail {

class BufferBuilder;

class Transform {
 public:
  virtual ~Transform() = default;

  [[nodiscard]] virtual Domain transform(const Domain& input) const = 0;

  [[nodiscard]] virtual Restrictions convert(Restrictions restrictions,
                                             bool forbid_fake_dim) const = 0;

  [[nodiscard]] virtual DomainPoint convert_color(DomainPoint color) const             = 0;
  [[nodiscard]] virtual DomainPoint convert_color_shape(DomainPoint color_shape) const = 0;
  [[nodiscard]] virtual DomainPoint convert_point(DomainPoint point) const             = 0;
  [[nodiscard]] virtual DomainPoint convert_extents(DomainPoint extents) const         = 0;
  /**
   * @brief Converts the per-dimension memory strides of the underlying storage to the strides
   * of the (possibly higher/lower-dimensional) transformed store.
   *
   * This mirrors `convert_extents` but operates on strides: dimensions that the transform adds
   * to the store but that are not backed by the storage (e.g. promoted or broadcast dimensions)
   * receive a stride of 0, while dimensions the transform removes from the storage are dropped.
   *
   * @param strides The storage's per-dimension strides.
   *
   * @return The strides expressed over the transformed store's dimensions.
   */
  [[nodiscard]] virtual SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_strides(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides) const = 0;

  [[nodiscard]] virtual SymbolicPoint invert(SymbolicPoint point) const      = 0;
  [[nodiscard]] virtual Restrictions invert(Restrictions restrictions) const = 0;

  [[nodiscard]] virtual DomainPoint invert_color(DomainPoint color) const             = 0;
  [[nodiscard]] virtual DomainPoint invert_color_shape(DomainPoint color_shape) const = 0;
  [[nodiscard]] virtual DomainPoint invert_point(DomainPoint point) const             = 0;
  [[nodiscard]] virtual DomainPoint invert_extents(DomainPoint extents) const         = 0;

  virtual void pack(BufferBuilder& buffer) const = 0;
  virtual void print(std::ostream& out) const    = 0;

  /**
   * @brief Applies the inverse transform (based on the derived class) to a tuple of
   * integers representing dimensions. For example, Transpose logically reorders the dims,
   * so Transpose::invert_dims() will undo the reordering.
   *
   * @param dims a tuple of integer dimensions
   *        with ids [0..dims.size()) in an arbitrary order.
   *
   * @returns a tuple of dims obtained
   *          by applying the inverse transform (based on the derived class).
   */
  [[nodiscard]] virtual SmallVector<std::int32_t, LEGATE_MAX_DIM> invert_dims(
    SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const = 0;
};

std::ostream& operator<<(std::ostream& out, const Transform& transform);

}  // namespace legate::detail
