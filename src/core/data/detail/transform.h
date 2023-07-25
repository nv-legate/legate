/* Copyright 2021-2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include <memory>

#include "core/data/shape.h"
#include "core/partitioning/restriction.h"
#include "core/runtime/detail/projection.h"
#include "core/utilities/detail/buffer_builder.h"
#include "core/utilities/typedefs.h"

namespace legate {
class Partition;
}  // namespace legate

namespace legate::detail {

class NonInvertibleTransformation : public std::exception {
 public:
  NonInvertibleTransformation() : error_message_("Non-invertible transformation") {}
  NonInvertibleTransformation(const std::string& error_message) : error_message_(error_message) {}
  const char* what() const throw() { return error_message_.c_str(); }

 private:
  std::string error_message_;
};

struct Transform {
  virtual Domain transform(const Domain& input) const                           = 0;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const = 0;
  virtual std::unique_ptr<Partition> convert(const Partition* partition) const  = 0;
  virtual std::unique_ptr<Partition> invert(const Partition* partition) const   = 0;
  virtual proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const    = 0;
  virtual Restrictions convert(const Restrictions& restrictions) const          = 0;
  virtual Restrictions invert(const Restrictions& restrictions) const           = 0;
  virtual Shape invert_extents(const Shape& extents) const                      = 0;
  virtual Shape invert_point(const Shape& point) const                          = 0;
  virtual bool is_convertible() const                                           = 0;
  virtual void pack(BufferBuilder& buffer) const                                = 0;
  virtual void print(std::ostream& out) const                                   = 0;
};

struct StoreTransform : public Transform {
  virtual ~StoreTransform() {}
  virtual int32_t target_ndim(int32_t source_ndim) const        = 0;
  virtual void find_imaginary_dims(std::vector<int32_t>&) const = 0;
};

struct TransformStack : public Transform, std::enable_shared_from_this<TransformStack> {
 public:
  TransformStack() {}
  TransformStack(std::unique_ptr<StoreTransform>&& transform,
                 const std::shared_ptr<TransformStack>& parent);
  TransformStack(std::unique_ptr<StoreTransform>&& transform,
                 std::shared_ptr<TransformStack>&& parent);

 public:
  Domain transform(const Domain& input) const override;
  Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  std::unique_ptr<Partition> convert(const Partition* partition) const override;
  std::unique_ptr<Partition> invert(const Partition* partition) const override;
  proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const override;
  Restrictions convert(const Restrictions& restrictions) const override;
  Restrictions invert(const Restrictions& restrictions) const override;
  Shape invert_extents(const Shape& extents) const override;
  Shape invert_point(const Shape& point) const override;
  bool is_convertible() const override { return convertible_; }
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

 public:
  std::unique_ptr<StoreTransform> pop();
  std::shared_ptr<TransformStack> push(std::unique_ptr<StoreTransform>&& transform);
  bool identity() const { return nullptr == transform_; }

 public:
  void dump() const;

 public:
  std::vector<int32_t> find_imaginary_dims() const;

 private:
  std::unique_ptr<StoreTransform> transform_{nullptr};
  std::shared_ptr<TransformStack> parent_{nullptr};
  bool convertible_{true};
};

class Shift : public StoreTransform {
 public:
  Shift(int32_t dim, int64_t offset);

 public:
  Domain transform(const Domain& input) const override;
  Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  std::unique_ptr<Partition> convert(const Partition* partition) const override;
  std::unique_ptr<Partition> invert(const Partition* partition) const override;
  proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const override;
  Restrictions convert(const Restrictions& restrictions) const override;
  Restrictions invert(const Restrictions& restrictions) const override;
  Shape invert_extents(const Shape& extents) const override;
  Shape invert_point(const Shape& point) const override;
  bool is_convertible() const override { return true; }
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

 public:
  virtual int32_t target_ndim(int32_t source_ndim) const override;

 public:
  virtual void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  int32_t dim_;
  int64_t offset_;
};

class Promote : public StoreTransform {
 public:
  Promote(int32_t extra_dim, int64_t dim_size);

 public:
  Domain transform(const Domain& input) const override;
  Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  std::unique_ptr<Partition> convert(const Partition* partition) const override;
  std::unique_ptr<Partition> invert(const Partition* partition) const override;
  proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const override;
  Restrictions convert(const Restrictions& restrictions) const override;
  Restrictions invert(const Restrictions& restrictions) const override;
  Shape invert_extents(const Shape& extents) const override;
  Shape invert_point(const Shape& point) const override;
  bool is_convertible() const override { return true; }
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

 public:
  virtual int32_t target_ndim(int32_t source_ndim) const override;

 public:
  virtual void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  int32_t extra_dim_;
  int64_t dim_size_;
};

class Project : public StoreTransform {
 public:
  Project(int32_t dim, int64_t coord);
  virtual ~Project() {}

 public:
  Domain transform(const Domain& domain) const override;
  Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  std::unique_ptr<Partition> convert(const Partition* partition) const override;
  std::unique_ptr<Partition> invert(const Partition* partition) const override;
  proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const override;
  Restrictions convert(const Restrictions& restrictions) const override;
  Restrictions invert(const Restrictions& restrictions) const override;
  Shape invert_extents(const Shape& extents) const override;
  Shape invert_point(const Shape& point) const override;
  bool is_convertible() const override { return true; }
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

 public:
  virtual int32_t target_ndim(int32_t source_ndim) const override;

 public:
  virtual void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  int32_t dim_;
  int64_t coord_;
};

class Transpose : public StoreTransform {
 public:
  Transpose(std::vector<int32_t>&& axes);

 public:
  virtual Domain transform(const Domain& domain) const override;
  virtual Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  virtual std::unique_ptr<Partition> convert(const Partition* partition) const override;
  virtual std::unique_ptr<Partition> invert(const Partition* partition) const override;
  virtual proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const override;
  virtual Restrictions convert(const Restrictions& restrictions) const override;
  virtual Restrictions invert(const Restrictions& restrictions) const override;
  virtual Shape invert_extents(const Shape& extents) const override;
  virtual Shape invert_point(const Shape& point) const override;
  virtual bool is_convertible() const override { return true; }
  virtual void pack(BufferBuilder& buffer) const override;
  virtual void print(std::ostream& out) const override;

 public:
  virtual int32_t target_ndim(int32_t source_ndim) const override;

 public:
  virtual void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  std::vector<int32_t> axes_;
  std::vector<int32_t> inverse_;
};

class Delinearize : public StoreTransform {
 public:
  Delinearize(int32_t dim, std::vector<int64_t>&& sizes);

 public:
  Domain transform(const Domain& domain) const override;
  Legion::DomainAffineTransform inverse_transform(int32_t in_dim) const override;
  std::unique_ptr<Partition> convert(const Partition* partition) const override;
  std::unique_ptr<Partition> invert(const Partition* partition) const override;
  proj::SymbolicPoint invert(const proj::SymbolicPoint& point) const override;
  Restrictions convert(const Restrictions& restrictions) const override;
  Restrictions invert(const Restrictions& restrictions) const override;
  Shape invert_extents(const Shape& extents) const override;
  Shape invert_point(const Shape& point) const override;
  bool is_convertible() const override { return false; }
  void pack(BufferBuilder& buffer) const override;
  void print(std::ostream& out) const override;

 public:
  virtual int32_t target_ndim(int32_t source_ndim) const override;

 public:
  virtual void find_imaginary_dims(std::vector<int32_t>&) const override;

 private:
  int32_t dim_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  int64_t volume_;
};

std::ostream& operator<<(std::ostream& out, const Transform& transform);

}  // namespace legate::detail
