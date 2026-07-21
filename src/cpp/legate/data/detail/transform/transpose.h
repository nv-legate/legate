/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/transform/store_transform.h>
#include <legate/partitioning/detail/restriction.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <iosfwd>

namespace legate::detail {

class BufferBuilder;

class Transpose final : public StoreTransform {
 public:
  explicit Transpose(SmallVector<std::int32_t, LEGATE_MAX_DIM>&& axes);

  [[nodiscard]] Domain transform(const Domain& domain) const override;

  [[nodiscard]] Restrictions convert(Restrictions restrictions,
                                     bool forbid_fake_dim) const override;

  [[nodiscard]] DomainPoint convert_color(DomainPoint color) const override;
  [[nodiscard]] DomainPoint convert_color_shape(DomainPoint color_shape) const override;
  [[nodiscard]] DomainPoint convert_point(DomainPoint point) const override;
  [[nodiscard]] DomainPoint convert_extents(DomainPoint extents) const override;
  [[nodiscard]] DomainPoint convert_strides(DomainPoint strides) const override;

  [[nodiscard]] SymbolicPoint invert(SymbolicPoint point) const override;
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;

  [[nodiscard]] DomainPoint invert_color(DomainPoint color) const override;
  [[nodiscard]] DomainPoint invert_color_shape(DomainPoint color_shape) const override;
  [[nodiscard]] DomainPoint invert_point(DomainPoint point) const override;
  [[nodiscard]] DomainPoint invert_extents(DomainPoint extents) const override;

  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] std::int32_t target_ndim(std::int32_t source_ndim) const override;

  void find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>&) const override;

  /**
   * @brief Reorder the input tuple of dimensions in the inverse order of
   * transpose, i.e., permute based on `inverse_` vector.
   *
   * @param dims a tuple of integer dimensions
   *        with ids [0..dims.size()) in an arbitrary order.
   *
   * @returns a tuple of dimensions permuted based on `inverse_`.
   */
  [[nodiscard]] SmallVector<std::int32_t, LEGATE_MAX_DIM> invert_dims(
    SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const override;

 private:
  SmallVector<std::int32_t, LEGATE_MAX_DIM> axes_{};
  SmallVector<std::int32_t, LEGATE_MAX_DIM> inverse_{};
};

}  // namespace legate::detail

#include <legate/data/detail/transform/transpose.inl>
