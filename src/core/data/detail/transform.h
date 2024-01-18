/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "core/data/shape.h"
#include "core/partitioning/restriction.h"
#include "core/runtime/detail/projection.h"
#include "core/utilities/detail/buffer_builder.h"
#include "core/utilities/internal_shared_ptr.h"
#include "core/utilities/typedefs.h"

#include <exception>
#include <iosfwd>
#include <memory>
#include <string>

namespace legate {
struct Partition;
}  // namespace legate

namespace legate::detail {

class NonInvertibleTransformation final : public std::exception {
 public:
  explicit NonInvertibleTransformation(std::string error_message = "Non-invertible transformation");

  [[nodiscard]] const char* what() const noexcept override;

 private:
  std::string error_message_{};
};

struct Transform {
  virtual ~Transform()                                              = default;
  [[nodiscard]] virtual Domain transform(const Domain& input) const = 0;
  [[nodiscard]] virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const = 0;
  [[nodiscard]] virtual std::unique_ptr<Partition> convert(const Partition* partition) const  = 0;
  [[nodiscard]] virtual std::unique_ptr<Partition> invert(const Partition* partition) const   = 0;
  [[nodiscard]] virtual proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const    = 0;
  [[nodiscard]] virtual Restrictions convert(const Restrictions& restrictions,
                                             bool forbid_fake_dim) const                      = 0;
  [[nodiscard]] virtual Restrictions invert(const Restrictions& restrictions) const           = 0;
  [[nodiscard]] virtual tuple<uint64_t> invert_color(tuple<uint64_t> color) const             = 0;
  [[nodiscard]] virtual tuple<uint64_t> invert_extents(const tuple<uint64_t>& extents) const  = 0;
  [[nodiscard]] virtual tuple<uint64_t> invert_point(const tuple<uint64_t>& point) const      = 0;
  [[nodiscard]] virtual bool is_convertible() const                                           = 0;
  virtual void pack(BufferBuilder& buffer) const                                              = 0;
  virtual void print(std::ostream& out) const                                                 = 0;
};

struct StoreTransform : public Transform {
  [[nodiscard]] virtual int32_t target_ndim(int32_t source_ndim) const = 0;
  virtual void find_imaginary_dims(std::vector<int32_t>&) const        = 0;
};

struct TransformStack final : public Transform {
 public:
  TransformStack() = default;
  TransformStack(std::unique_ptr<StoreTransform>&& transform,
                 const InternalSharedPtr<TransformStack>& parent);
  TransformStack(std::unique_ptr<StoreTransform>&& transform,
                 InternalSharedPtr<TransformStack>&& parent);

  [[nodiscard]] Domain transform(const Domain& input) const override;
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  [[nodiscard]] std::unique_ptr<Partition> convert(const Partition* partition) const override;
  [[nodiscard]] std::unique_ptr<Partition> invert(const Partition* partition) const override;
  [[nodiscard]] proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const override;
  [[nodiscard]] Restrictions convert(const Restrictions& restrictions,
                                     bool forbid_fake_dim) const override;
  [[nodiscard]] Restrictions invert(const Restrictions& restrictions) const override;
  [[nodiscard]] tuple<uint64_t> invert_color(tuple<uint64_t> color) const override;
  [[nodiscard]] tuple<uint64_t> invert_extents(const tuple<uint64_t>& extents) const override;
  [[nodiscard]] tuple<uint64_t> invert_point(const tuple<uint64_t>& point) const override;
  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] std::unique_ptr<StoreTransform> pop();
  [[nodiscard]] bool identity() const;

  void dump() const;

  [[nodiscard]] std::vector<int32_t> find_imaginary_dims() const;

 private:
  struct private_tag {};

  TransformStack(private_tag,
                 std::unique_ptr<StoreTransform>&& transform,
                 InternalSharedPtr<TransformStack> parent);

  std::unique_ptr<StoreTransform> transform_{};
  InternalSharedPtr<TransformStack> parent_{};
  bool convertible_{true};
};

class Shift final : public StoreTransform {
 public:
  Shift(int32_t dim, int64_t offset);

  [[nodiscard]] Domain transform(const Domain& input) const override;
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  [[nodiscard]] std::unique_ptr<Partition> convert(const Partition* partition) const override;
  [[nodiscard]] std::unique_ptr<Partition> invert(const Partition* partition) const override;
  [[nodiscard]] proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const override;
  [[nodiscard]] Restrictions convert(const Restrictions& restrictions,
                                     bool forbid_fake_dim) const override;
  [[nodiscard]] Restrictions invert(const Restrictions& restrictions) const override;
  [[nodiscard]] tuple<uint64_t> invert_color(tuple<uint64_t> color) const override;
  [[nodiscard]] tuple<uint64_t> invert_extents(const tuple<uint64_t>& extents) const override;
  [[nodiscard]] tuple<uint64_t> invert_point(const tuple<uint64_t>& point) const override;
  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] int32_t target_ndim(int32_t source_ndim) const override;

  void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  int32_t dim_{};
  int64_t offset_{};
};

class Promote final : public StoreTransform {
 public:
  Promote(int32_t extra_dim, int64_t dim_size);

  [[nodiscard]] Domain transform(const Domain& input) const override;
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  [[nodiscard]] std::unique_ptr<Partition> convert(const Partition* partition) const override;
  [[nodiscard]] std::unique_ptr<Partition> invert(const Partition* partition) const override;
  [[nodiscard]] proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const override;
  [[nodiscard]] Restrictions convert(const Restrictions& restrictions,
                                     bool forbid_fake_dim) const override;
  [[nodiscard]] Restrictions invert(const Restrictions& restrictions) const override;
  [[nodiscard]] tuple<uint64_t> invert_color(tuple<uint64_t> color) const override;
  [[nodiscard]] tuple<uint64_t> invert_extents(const tuple<uint64_t>& extents) const override;
  [[nodiscard]] tuple<uint64_t> invert_point(const tuple<uint64_t>& point) const override;
  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] int32_t target_ndim(int32_t source_ndim) const override;

  void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  int32_t extra_dim_{};
  int64_t dim_size_{};
};

class Project final : public StoreTransform {
 public:
  Project(int32_t dim, int64_t coord);

  [[nodiscard]] Domain transform(const Domain& input) const override;
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  [[nodiscard]] std::unique_ptr<Partition> convert(const Partition* partition) const override;
  [[nodiscard]] std::unique_ptr<Partition> invert(const Partition* partition) const override;
  [[nodiscard]] proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const override;
  [[nodiscard]] Restrictions convert(const Restrictions& restrictions,
                                     bool forbid_fake_dim) const override;
  [[nodiscard]] Restrictions invert(const Restrictions& restrictions) const override;
  [[nodiscard]] tuple<uint64_t> invert_color(tuple<uint64_t> color) const override;
  [[nodiscard]] tuple<uint64_t> invert_extents(const tuple<uint64_t>& extents) const override;
  [[nodiscard]] tuple<uint64_t> invert_point(const tuple<uint64_t>& point) const override;
  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] int32_t target_ndim(int32_t source_ndim) const override;

  void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  int32_t dim_{};
  int64_t coord_{};
};

class Transpose final : public StoreTransform {
 public:
  explicit Transpose(std::vector<int32_t>&& axes);

  [[nodiscard]] Domain transform(const Domain& domain) const override;
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  [[nodiscard]] std::unique_ptr<Partition> convert(const Partition* partition) const override;
  [[nodiscard]] std::unique_ptr<Partition> invert(const Partition* partition) const override;
  [[nodiscard]] proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const override;
  [[nodiscard]] Restrictions convert(const Restrictions& restrictions,
                                     bool forbid_fake_dim) const override;
  [[nodiscard]] Restrictions invert(const Restrictions& restrictions) const override;
  [[nodiscard]] tuple<uint64_t> invert_color(tuple<uint64_t> color) const override;
  [[nodiscard]] tuple<uint64_t> invert_extents(const tuple<uint64_t>& extents) const override;
  [[nodiscard]] tuple<uint64_t> invert_point(const tuple<uint64_t>& point) const override;
  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] int32_t target_ndim(int32_t source_ndim) const override;

  void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  std::vector<int32_t> axes_{};
  std::vector<int32_t> inverse_{};
};

class Delinearize final : public StoreTransform {
 public:
  Delinearize(int32_t dim, std::vector<uint64_t>&& sizes);

  [[nodiscard]] Domain transform(const Domain& domain) const override;
  [[nodiscard]] Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  [[nodiscard]] std::unique_ptr<Partition> convert(const Partition* partition) const override;
  [[nodiscard]] std::unique_ptr<Partition> invert(const Partition* partition) const override;
  [[nodiscard]] proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const override;
  [[nodiscard]] Restrictions convert(const Restrictions& restrictions,
                                     bool forbid_fake_dim) const override;
  [[nodiscard]] Restrictions invert(const Restrictions& restrictions) const override;
  [[nodiscard]] tuple<uint64_t> invert_color(tuple<uint64_t> color) const override;
  [[nodiscard]] tuple<uint64_t> invert_extents(const tuple<uint64_t>& extents) const override;
  [[nodiscard]] tuple<uint64_t> invert_point(const tuple<uint64_t>& point) const override;
  [[nodiscard]] bool is_convertible() const override;
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

  [[nodiscard]] int32_t target_ndim(int32_t source_ndim) const override;

  void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  int32_t dim_{};
  std::vector<uint64_t> sizes_{};
  std::vector<uint64_t> strides_{};
  uint64_t volume_{};
};

std::ostream& operator<<(std::ostream& out, const Transform& transform);

}  // namespace legate::detail

#include "core/data/detail/transform.inl"
