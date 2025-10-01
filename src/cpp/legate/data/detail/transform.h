/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/shape.h>
#include <legate/partitioning/detail/restriction.h>
#include <legate/runtime/detail/projection.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace legate::detail {

class LEGATE_EXPORT NonInvertibleTransformation : public std::runtime_error {
 public:
  explicit NonInvertibleTransformation(
    std::string_view error_message = "Non-invertible transformation");
};

class Transform {
 public:
  virtual ~Transform()                                              = default;
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

class StoreTransform : public Transform {
 public:
  [[nodiscard]] virtual std::int32_t target_ndim(std::int32_t source_ndim) const     = 0;
  virtual void find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>&) const = 0;
};

class TransformStack final : public Transform {
 public:
  TransformStack() = default;
  TransformStack(std::unique_ptr<StoreTransform>&& transform,
                 const InternalSharedPtr<TransformStack>& parent);
  TransformStack(std::unique_ptr<StoreTransform>&& transform,
                 InternalSharedPtr<TransformStack>&& parent);

  [[nodiscard]] Domain transform(const Domain& input) const override;
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(std::int32_t in_dim) const override;

  [[nodiscard]] Restrictions convert(Restrictions restrictions,
                                     bool forbid_fake_dim) const override;

  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const override;
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> convert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const override;

  [[nodiscard]] proj::SymbolicPoint invert(proj::SymbolicPoint point) const override;
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;

  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const override;
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> invert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const override;

  [[nodiscard]] bool is_convertible() const override;
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
  bool convertible_{true};
};

class Shift final : public StoreTransform {
 public:
  Shift(std::int32_t dim, std::int64_t offset);

  [[nodiscard]] Domain transform(const Domain& input) const override;
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(std::int32_t in_dim) const override;

  [[nodiscard]] Restrictions convert(Restrictions restrictions,
                                     bool forbid_fake_dim) const override;

  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const override;
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> convert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const override;

  [[nodiscard]] proj::SymbolicPoint invert(proj::SymbolicPoint point) const override;
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;

  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const override;
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> invert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const override;

  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] std::int32_t target_ndim(std::int32_t source_ndim) const override;

  void find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>&) const override;

  /**
   * @brief Is a NOOP for Shift.
   *
   * @param dims a tuple of integer dimensions
   *        with ids [0..dims.size()) in an arbitrary order.
   *
   * @returns dims unmodified.
   */
  [[nodiscard]] SmallVector<std::int32_t, LEGATE_MAX_DIM> invert_dims(
    SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const override;

 private:
  std::int32_t dim_{};
  std::int64_t offset_{};
};

class Promote final : public StoreTransform {
 public:
  Promote(std::int32_t extra_dim, std::int64_t dim_size);

  [[nodiscard]] Domain transform(const Domain& input) const override;
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(std::int32_t in_dim) const override;

  [[nodiscard]] Restrictions convert(Restrictions restrictions,
                                     bool forbid_fake_dim) const override;

  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const override;
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> convert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const override;

  [[nodiscard]] proj::SymbolicPoint invert(proj::SymbolicPoint point) const override;
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;

  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const override;
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> invert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const override;

  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] std::int32_t target_ndim(std::int32_t source_ndim) const override;

  void find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>&) const override;

  /**
   * @brief Remove the value that equals `extra_dim_` from input tuple of
   * dimensions.
   *
   * @param dims a tuple of integer dimensions
   *        with ids [0..dims.size()) in an arbitrary order.
   *
   * @returns a tuple of dimensions that has `extra_dim_` removed from it.
   */
  [[nodiscard]] SmallVector<std::int32_t, LEGATE_MAX_DIM> invert_dims(
    SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const override;

 private:
  std::int32_t extra_dim_{};
  std::int64_t dim_size_{};
};

class Project final : public StoreTransform {
 public:
  Project(std::int32_t dim, std::int64_t coord);

  [[nodiscard]] Domain transform(const Domain& input) const override;
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(std::int32_t in_dim) const override;

  [[nodiscard]] Restrictions convert(Restrictions restrictions,
                                     bool forbid_fake_dim) const override;

  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const override;
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> convert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const override;

  [[nodiscard]] proj::SymbolicPoint invert(proj::SymbolicPoint point) const override;
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;

  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const override;
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> invert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const override;

  [[nodiscard]] bool is_convertible() const override;
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
  [[nodiscard]] Domain transform(const Domain& input) const override;
  /**
   * @brief Construct an inverse transform matrix for this `DimBroadcast`.
   *
   * The inverse transform matrix is an `in_dim`x`in_dim` diagonal matrix where the `dim_`-th
   * diagonal element is 0 and all the other diagonal elements are 1.
   *
   * @param in_dim The number of dimensions of an input
   *
   * @return A `Legion::DomainAffineTransform` that implements this broadcasting
   */
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(std::int32_t in_dim) const override;

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
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const override;
  /**
   * @brief Convert a color shape.
   *
   * No-op for this transformation.
   *
   * @return Unmodified `color_shape`
   */
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const override;
  /**
   * @brief Convert a point.
   *
   * No-op for this transformation.
   *
   * @return Unmodified `point`
   */
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> convert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const override;
  /**
   * @brief Convert extents.
   *
   * The extent of dimension `dim_` is updated to `dim_size_`.
   *
   * @return `extents` with the `dim_`-th extent updated to `dim_size_`.
   */
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const override;
  /**
   * @brief Invert a symbolic point.
   *
   * The `dim_`-th coordinate in the input symbolic point is ignored.
   *
   * @return `point` with the `dim_`-th expression updated to `legate::constant(0)`.
   */
  [[nodiscard]] proj::SymbolicPoint invert(proj::SymbolicPoint point) const override;
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
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const override;
  /**
   * @brief Invert a color shape.
   *
   * The `dim_`-th size is set to 1.
   *
   * @return `color_shape` with the `dim_`-th size updated to 0.
   */
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const override;
  /**
   * @brief Invert a point.
   *
   * The `dim_`-th coordinate is set to 0.
   *
   * @return `point` with the `dim_`-th coordinate updated to 0.
   */
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> invert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const override;
  /**
   * @brief Invert extents.
   *
   * The `dim_`-th extent is set to 1.
   *
   * @return `extents` with the `dim_`-th extent updated to 1.
   */
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const override;

  /**
   * @brief Always return true, as `DimBroadcast` is convertible.
   */
  [[nodiscard]] bool is_convertible() const override;
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

class Transpose final : public StoreTransform {
 public:
  explicit Transpose(SmallVector<std::int32_t, LEGATE_MAX_DIM>&& axes);

  [[nodiscard]] Domain transform(const Domain& domain) const override;
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(std::int32_t in_dim) const override;

  [[nodiscard]] Restrictions convert(Restrictions restrictions,
                                     bool forbid_fake_dim) const override;

  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const override;
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> convert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const override;

  [[nodiscard]] proj::SymbolicPoint invert(proj::SymbolicPoint point) const override;
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;

  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const override;
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> invert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const override;

  [[nodiscard]] bool is_convertible() const override;
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

class Delinearize final : public StoreTransform {
 public:
  Delinearize(std::int32_t dim, SmallVector<std::uint64_t, LEGATE_MAX_DIM>&& sizes);

  [[nodiscard]] Domain transform(const Domain& domain) const override;
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(std::int32_t in_dim) const override;

  [[nodiscard]] Restrictions convert(Restrictions restrictions,
                                     bool forbid_fake_dim) const override;

  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const override;
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> convert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> convert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const override;

  [[nodiscard]] proj::SymbolicPoint invert(proj::SymbolicPoint point) const override;
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;

  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_color_shape(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const override;
  [[nodiscard]] SmallVector<std::int64_t, LEGATE_MAX_DIM> invert_point(
    SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const override;
  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> invert_extents(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const override;

  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] std::int32_t target_ndim(std::int32_t source_ndim) const override;

  void find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>&) const override;

  /**
   * @brief Remove the new dimensions that were added as a result of Deliearize,
   * and add back the original dimension, while also renumbering the higher
   * dimensions.
   *
   * @param dims a tuple of integer dimensions
   *        with ids [0..dims.size()) in an arbitrary order.
   *
   * @returns a tuple of dimensions with values in the range
   *          (dim_, x=(dim_ + sizes_.size()) removed and all values y >= x
   *          renumbered to y-x-1.
   */
  [[nodiscard]] SmallVector<std::int32_t, LEGATE_MAX_DIM> invert_dims(
    SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const override;

 private:
  std::int32_t dim_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> sizes_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> strides_{};
  std::uint64_t volume_{};
};

std::ostream& operator<<(std::ostream& out, const Transform& transform);

}  // namespace legate::detail

#include <legate/data/detail/transform.inl>
