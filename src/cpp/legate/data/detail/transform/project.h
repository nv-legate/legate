/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/transform/store_transform.h>
#include <legate/geometry/point.h>
#include <legate/geometry/rect.h>
#include <legate/partitioning/detail/restriction.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <iosfwd>

namespace legate::detail {

class BufferBuilder;

class Project final : public StoreTransform {
 public:
  Project(std::int32_t dim, std::int64_t coord);

  [[nodiscard]] Rect transform(const Rect& input) const override;

  [[nodiscard]] Restrictions convert(Restrictions restrictions,
                                     bool forbid_fake_dim) const override;

  [[nodiscard]] Point convert_color(Point color) const override;
  [[nodiscard]] Point convert_color_shape(Point color_shape) const override;
  [[nodiscard]] Point convert_point(Point point) const override;
  [[nodiscard]] Point convert_extents(Point extents) const override;
  [[nodiscard]] Point convert_strides(Point strides) const override;

  [[nodiscard]] SymbolicPoint invert(SymbolicPoint point) const override;
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;

  [[nodiscard]] Point invert_color(Point color) const override;
  [[nodiscard]] Point invert_color_shape(Point color_shape) const override;
  [[nodiscard]] Point invert_point(Point point) const override;
  [[nodiscard]] Point invert_extents(Point extents) const override;

  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] std::int32_t target_ndim(std::int32_t source_ndim) const override;

  void find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>&) const override;

  /**
   * @brief Add back `dim_` to the input tuple of
   * dimensions at position `dim_`.
   *
   * @param dims a tuple of integer dimensions
   *        with ids [0..dims.size()) in an arbitrary order.
   *
   * @returns a tuple of dimensions with `dim_` inserted back.
   */
  [[nodiscard]] SmallVector<std::int32_t, LEGATE_MAX_DIM> invert_dims(
    SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const override;

 private:
  std::int32_t dim_{};
  std::int64_t coord_{};
};

}  // namespace legate::detail

#include <legate/data/detail/transform/project.inl>
