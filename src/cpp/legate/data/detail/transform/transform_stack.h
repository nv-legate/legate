/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/transform/store_transform.h>
#include <legate/data/detail/transform/transform.h>
#include <legate/partitioning/detail/restriction.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <iosfwd>
#include <memory>

namespace legate::detail {

class BufferBuilder;

class TransformStack final : public Transform {
 public:
  TransformStack() = default;
  TransformStack(std::unique_ptr<StoreTransform>&& transform,
                 const InternalSharedPtr<TransformStack>& parent);
  TransformStack(std::unique_ptr<StoreTransform>&& transform,
                 InternalSharedPtr<TransformStack>&& parent);

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

  [[nodiscard]] std::unique_ptr<StoreTransform> pop();
  [[nodiscard]] bool identity() const;

  void dump() const;

  [[nodiscard]] SmallVector<std::int32_t, LEGATE_MAX_DIM> find_imaginary_dims() const;

  /**
   * @brief Invokes `invert_dims()` down the transform stack recursively.
   *
   * @param dims a tuple of integer dimensions
   *        with ids [0..dims.size()) in an arbitrary order.
   *
   * @returns a tuple of dims obtained
   *          by applying `invert_dims()` recursively down the transform stack.
   */
  [[nodiscard]] SmallVector<std::int32_t, LEGATE_MAX_DIM> invert_dims(
    SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const override;

 private:
  struct private_tag {};

  TransformStack(private_tag,
                 std::unique_ptr<StoreTransform>&& transform,
                 InternalSharedPtr<TransformStack> parent);

  template <typename VISITOR, typename T>
  auto convert_(VISITOR visitor, T&& input) const;

  template <typename VISITOR, typename T>
  auto invert_(VISITOR visitor, T&& input) const;

  std::unique_ptr<StoreTransform> transform_{};
  InternalSharedPtr<TransformStack> parent_{};
};

}  // namespace legate::detail

#include <legate/data/detail/transform/transform_stack.inl>
