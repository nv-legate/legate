/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "legate/data/shape.h"
#include "legate/partitioning/detail/restriction.h"
#include "legate/runtime/detail/projection.h"
#include "legate/utilities/detail/buffer_builder.h"
#include "legate/utilities/internal_shared_ptr.h"
#include "legate/utilities/typedefs.h"

#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace legate::detail {

class NonInvertibleTransformation : public std::runtime_error {
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

  [[nodiscard]] virtual tuple<std::uint64_t> convert_color(tuple<std::uint64_t> color) const = 0;
  [[nodiscard]] virtual tuple<std::uint64_t> convert_color_shape(
    tuple<std::uint64_t> extents) const                                                    = 0;
  [[nodiscard]] virtual tuple<std::int64_t> convert_point(tuple<std::int64_t> point) const = 0;
  [[nodiscard]] virtual tuple<std::uint64_t> convert_extents(
    tuple<std::uint64_t> extents) const = 0;

  [[nodiscard]] virtual proj::SymbolicPoint invert(proj::SymbolicPoint point) const = 0;
  [[nodiscard]] virtual Restrictions invert(Restrictions restrictions) const        = 0;

  [[nodiscard]] virtual tuple<std::uint64_t> invert_color(tuple<std::uint64_t> color) const = 0;
  [[nodiscard]] virtual tuple<std::uint64_t> invert_color_shape(
    tuple<std::uint64_t> extents) const                                                         = 0;
  [[nodiscard]] virtual tuple<std::int64_t> invert_point(tuple<std::int64_t> point) const       = 0;
  [[nodiscard]] virtual tuple<std::uint64_t> invert_extents(tuple<std::uint64_t> extents) const = 0;

  [[nodiscard]] virtual bool is_convertible() const = 0;
  virtual void pack(BufferBuilder& buffer) const    = 0;
  virtual void print(std::ostream& out) const       = 0;
};

class StoreTransform : public Transform {
 public:
  [[nodiscard]] virtual std::int32_t target_ndim(std::int32_t source_ndim) const = 0;
  virtual void find_imaginary_dims(std::vector<std::int32_t>&) const             = 0;
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

  [[nodiscard]] tuple<std::uint64_t> convert_color(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::uint64_t> convert_color_shape(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::int64_t> convert_point(tuple<std::int64_t> point) const override;
  [[nodiscard]] tuple<std::uint64_t> convert_extents(tuple<std::uint64_t> extents) const override;

  [[nodiscard]] proj::SymbolicPoint invert(proj::SymbolicPoint point) const override;
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;

  [[nodiscard]] tuple<std::uint64_t> invert_color(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::uint64_t> invert_color_shape(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::int64_t> invert_point(tuple<std::int64_t> point) const override;
  [[nodiscard]] tuple<std::uint64_t> invert_extents(tuple<std::uint64_t> extents) const override;

  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] std::unique_ptr<StoreTransform> pop();
  [[nodiscard]] bool identity() const;

  void dump() const;

  [[nodiscard]] std::vector<std::int32_t> find_imaginary_dims() const;

 private:
  struct private_tag {};

  TransformStack(private_tag,
                 std::unique_ptr<StoreTransform>&& transform,
                 InternalSharedPtr<TransformStack> parent);

  template <typename VISITOR, typename T>
  decltype(auto) convert_(VISITOR visitor, T input) const;

  template <typename VISITOR, typename T>
  decltype(auto) invert_(VISITOR visitor, T input) const;

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

  [[nodiscard]] tuple<std::uint64_t> convert_color(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::uint64_t> convert_color_shape(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::int64_t> convert_point(tuple<std::int64_t> point) const override;
  [[nodiscard]] tuple<std::uint64_t> convert_extents(tuple<std::uint64_t> extents) const override;

  [[nodiscard]] proj::SymbolicPoint invert(proj::SymbolicPoint point) const override;
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;

  [[nodiscard]] tuple<std::uint64_t> invert_color(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::uint64_t> invert_color_shape(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::int64_t> invert_point(tuple<std::int64_t> point) const override;
  [[nodiscard]] tuple<std::uint64_t> invert_extents(tuple<std::uint64_t> extents) const override;

  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] std::int32_t target_ndim(std::int32_t source_ndim) const override;

  void find_imaginary_dims(std::vector<std::int32_t>&) const override;

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

  [[nodiscard]] tuple<std::uint64_t> convert_color(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::uint64_t> convert_color_shape(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::int64_t> convert_point(tuple<std::int64_t> point) const override;
  [[nodiscard]] tuple<std::uint64_t> convert_extents(tuple<std::uint64_t> extents) const override;

  [[nodiscard]] proj::SymbolicPoint invert(proj::SymbolicPoint point) const override;
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;

  [[nodiscard]] tuple<std::uint64_t> invert_color(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::uint64_t> invert_color_shape(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::int64_t> invert_point(tuple<std::int64_t> point) const override;
  [[nodiscard]] tuple<std::uint64_t> invert_extents(tuple<std::uint64_t> extents) const override;

  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] std::int32_t target_ndim(std::int32_t source_ndim) const override;

  void find_imaginary_dims(std::vector<std::int32_t>&) const override;

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

  [[nodiscard]] tuple<std::uint64_t> convert_color(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::uint64_t> convert_color_shape(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::int64_t> convert_point(tuple<std::int64_t> point) const override;
  [[nodiscard]] tuple<std::uint64_t> convert_extents(tuple<std::uint64_t> extents) const override;

  [[nodiscard]] proj::SymbolicPoint invert(proj::SymbolicPoint point) const override;
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;

  [[nodiscard]] tuple<std::uint64_t> invert_color(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::uint64_t> invert_color_shape(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::int64_t> invert_point(tuple<std::int64_t> point) const override;
  [[nodiscard]] tuple<std::uint64_t> invert_extents(tuple<std::uint64_t> extents) const override;

  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] std::int32_t target_ndim(std::int32_t source_ndim) const override;

  void find_imaginary_dims(std::vector<std::int32_t>&) const override;

 private:
  std::int32_t dim_{};
  std::int64_t coord_{};
};

class Transpose final : public StoreTransform {
 public:
  explicit Transpose(std::vector<std::int32_t>&& axes);

  [[nodiscard]] Domain transform(const Domain& domain) const override;
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(std::int32_t in_dim) const override;

  [[nodiscard]] Restrictions convert(Restrictions restrictions,
                                     bool forbid_fake_dim) const override;

  [[nodiscard]] tuple<std::uint64_t> convert_color(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::uint64_t> convert_color_shape(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::int64_t> convert_point(tuple<std::int64_t> point) const override;
  [[nodiscard]] tuple<std::uint64_t> convert_extents(tuple<std::uint64_t> extents) const override;

  [[nodiscard]] proj::SymbolicPoint invert(proj::SymbolicPoint point) const override;
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;

  [[nodiscard]] tuple<std::uint64_t> invert_color(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::uint64_t> invert_color_shape(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::int64_t> invert_point(tuple<std::int64_t> point) const override;
  [[nodiscard]] tuple<std::uint64_t> invert_extents(tuple<std::uint64_t> extents) const override;

  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] std::int32_t target_ndim(std::int32_t source_ndim) const override;

  void find_imaginary_dims(std::vector<std::int32_t>&) const override;

 private:
  std::vector<std::int32_t> axes_{};
  std::vector<std::int32_t> inverse_{};
};

class Delinearize final : public StoreTransform {
 public:
  Delinearize(std::int32_t dim, std::vector<std::uint64_t>&& sizes);

  [[nodiscard]] Domain transform(const Domain& domain) const override;
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(std::int32_t in_dim) const override;

  [[nodiscard]] Restrictions convert(Restrictions restrictions,
                                     bool forbid_fake_dim) const override;

  [[nodiscard]] tuple<std::uint64_t> convert_color(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::uint64_t> convert_color_shape(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::int64_t> convert_point(tuple<std::int64_t> point) const override;
  [[nodiscard]] tuple<std::uint64_t> convert_extents(tuple<std::uint64_t> extents) const override;

  [[nodiscard]] proj::SymbolicPoint invert(proj::SymbolicPoint point) const override;
  [[nodiscard]] Restrictions invert(Restrictions restrictions) const override;

  [[nodiscard]] tuple<std::uint64_t> invert_color(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::uint64_t> invert_color_shape(tuple<std::uint64_t> color) const override;
  [[nodiscard]] tuple<std::int64_t> invert_point(tuple<std::int64_t> point) const override;
  [[nodiscard]] tuple<std::uint64_t> invert_extents(tuple<std::uint64_t> extents) const override;

  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] std::int32_t target_ndim(std::int32_t source_ndim) const override;

  void find_imaginary_dims(std::vector<std::int32_t>&) const override;

 private:
  std::int32_t dim_{};
  std::vector<std::uint64_t> sizes_{};
  std::vector<std::uint64_t> strides_{};
  std::uint64_t volume_{};
};

std::ostream& operator<<(std::ostream& out, const Transform& transform);

}  // namespace legate::detail

#include "legate/data/detail/transform.inl"
