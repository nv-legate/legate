/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/restriction.h>
#include <legate/runtime/detail/projection.h>
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
  [[nodiscard]] virtual Legion::DomainAffineTransform inverse_transform(
    std::int32_t in_dim) const = 0;

  [[nodiscard]] virtual Restrictions convert(Restrictions restrictions,
                                             bool forbid_fake_dim) const = 0;

  [[nodiscard]] virtual SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const = 0;
  [[nodiscard]] virtual SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const = 0;
  [[nodiscard]] virtual SmallVector<std::int64_t, LEGATE_MAX_DIM> convert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const = 0;
  [[nodiscard]] virtual SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const = 0;

  [[nodiscard]] virtual proj::SymbolicPoint invert(proj::SymbolicPoint point) const = 0;
  [[nodiscard]] virtual Restrictions invert(Restrictions restrictions) const        = 0;

  [[nodiscard]] virtual SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const = 0;
  [[nodiscard]] virtual SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const = 0;
  [[nodiscard]] virtual SmallVector<std::int64_t, LEGATE_MAX_DIM> invert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const = 0;
  [[nodiscard]] virtual SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const = 0;

  [[nodiscard]] virtual bool is_convertible() const = 0;
  virtual void pack(BufferBuilder& buffer) const    = 0;
  virtual void print(std::ostream& out) const       = 0;

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
