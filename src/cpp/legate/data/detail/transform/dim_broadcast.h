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

/**
 * @brief Store transformation that logically broadcasts a unit-size dimension
 */
class DimBroadcast final : public StoreTransform {
 public:
  /**
   * @brief Construct a `DimBroadcast` transformation.
   *
   * @param dim A dimension to broadcast.
   * @param dim_size A dimension size after broadcasting.
   */
  DimBroadcast(std::int32_t dim, std::uint64_t dim_size);
  /**
   * @brief Broadcast dimension `dim_` by size `dim_size_`.
   *
   * @param input An input domain.
   *
   * @return A domain where dimension `dim_` has size `dim_size_` and the other dimensions have the
   * same size as the input.
   */
  [[nodiscard]] Rect transform(const Rect& input) const override;

  /**
   * @brief Convert `Restrictions`.
   *
   * The input `Restrictions` are updated so that partitioning on dimension `dim_` is either
   * forbidden or avoided.
   *
   * @param forbid_fake_dim If set true, set `Restriction::FORBID` to dimension `dim_`.
   *
   * @return A `Restrictions` object after conversion
   */
  [[nodiscard]] Restrictions convert(Restrictions restrictions,
                                     bool forbid_fake_dim) const override;
  /**
   * @brief Convert a color.
   *
   * No-op for this transformation.
   *
   * @return Unmodified `color`
   */
  [[nodiscard]] Point convert_color(Point color) const override;
  /**
   * @brief Convert a color shape.
   *
   * No-op for this transformation.
   *
   * @return Unmodified `color_shape`
   */
  [[nodiscard]] Point convert_color_shape(Point color_shape) const override;
  /**
   * @brief Convert a point.
   *
   * No-op for this transformation.
   *
   * @return Unmodified `point`
   */
  [[nodiscard]] Point convert_point(Point point) const override;
  /**
   * @brief Convert extents.
   *
   * The extent of dimension `dim_` is updated to `dim_size_`.
   *
   * @return `extents` with the `dim_`-th extent updated to `dim_size_`.
   */
  [[nodiscard]] Point convert_extents(Point extents) const override;
  /**
   * @brief Convert strides.
   *
   * The stride of the broadcast dimension `dim_` is set to 0 so that indexing along it always
   * refers to the single backing element.
   *
   * @return `strides` with the `dim_`-th stride updated to 0.
   */
  [[nodiscard]] Point convert_strides(Point strides) const override;
  /**
   * @brief Invert a symbolic point.
   *
   * The `dim_`-th coordinate in the input symbolic point is ignored.
   *
   * @return `point` with the `dim_`-th expression updated to `legate::constant(0)`.
   */
  [[nodiscard]] SymbolicPoint invert(SymbolicPoint point) const override;
  /**
   * @brief Invert `Restrictions`.
   *
   * No-op for this transformation.
   *
   * @return Unmodified `restrictions`
   */
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;
  /**
   * @brief Invert a color.
   *
   * The `dim_`-th coordinate is set to 0.
   *
   * @return `color` with the `dim_`-th coordinate updated to 0.
   */
  [[nodiscard]] Point invert_color(Point color) const override;
  /**
   * @brief Invert a color shape.
   *
   * The `dim_`-th size is set to 1.
   *
   * @return `color_shape` with the `dim_`-th size updated to 0.
   */
  [[nodiscard]] Point invert_color_shape(Point color_shape) const override;
  /**
   * @brief Invert a point.
   *
   * The `dim_`-th coordinate is set to 0.
   *
   * @return `point` with the `dim_`-th coordinate updated to 0.
   */
  [[nodiscard]] Point invert_point(Point point) const override;
  /**
   * @brief Invert extents.
   *
   * The `dim_`-th extent is set to 1.
   *
   * @return `extents` with the `dim_`-th extent updated to 1.
   */
  [[nodiscard]] Point invert_extents(Point extents) const override;

  /**
   * @brief Serialize this `DimBroadcast` into the passed `buffer`.
   */
  void pack(BufferBuilder& buffer) const override;
  /**
   * @brief Print a human-readable string of this `DimBroadcast` to the `out` stream.
   */
  void print(std::ostream& out) const override;

  /**
   * @brief Return the target number of dimensions.
   *
   * Same as the `source_ndim`.
   *
   * @return `source_ndim`.
   */
  [[nodiscard]] std::int32_t target_ndim(std::int32_t source_ndim) const override;
  /**
   * @brief Return imaginary dimensions.
   *
   * No-op for this transformation.
   */
  void find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>&) const override;
  /**
   * @brief Invert a dimension ordering
   *
   * No-op for this transformation.
   *
   * @return Unmodified `dims`.
   */
  [[nodiscard]] SmallVector<std::int32_t, LEGATE_MAX_DIM> invert_dims(
    SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const override;

 private:
  std::int32_t dim_{};
  std::uint64_t dim_size_{};
};

}  // namespace legate::detail

#include <legate/data/detail/transform/dim_broadcast.inl>
