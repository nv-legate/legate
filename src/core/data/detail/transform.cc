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

#include "core/data/detail/transform.h"

#include "core/partitioning/partition.h"
#include "core/utilities/detail/buffer_builder.h"

#include <algorithm>
#include <iostream>
#include <iterator>

namespace legate::detail {

std::unique_ptr<Partition> TransformStack::convert(const Partition* partition) const
{
  if (identity()) {
    return partition->clone();
  }

  if (parent_->identity()) {
    return transform_->convert(partition);
  }

  auto result = parent_->convert(partition);
  return transform_->convert(result.get());
}

std::unique_ptr<Partition> TransformStack::invert(const Partition* partition) const
{
  if (identity()) {
    return partition->clone();
  }

  auto result = transform_->invert(partition);
  if (parent_->identity()) {
    return result;
  }
  return parent_->invert(result.get());
}

proj::SymbolicPoint TransformStack::invert(const proj::SymbolicPoint& point) const
{
  if (identity()) {
    return point;
  }

  auto result = transform_->invert(point);
  if (parent_->identity()) {
    return result;
  }
  // FIXME: This is very inefficient if the stack is large. Since each invert() takes by const
  // reference and returns a fresh SymbolicPoint (i.e. std::vector), we end up with a bunch of
  // temporary objects and allocations. We should be std::move-ing this!
  return parent_->invert(result);
}

Restrictions TransformStack::convert(const Restrictions& restrictions, bool forbid_fake_dim) const
{
  if (identity()) {
    return restrictions;
  }
  if (parent_->identity()) {
    return transform_->convert(restrictions, forbid_fake_dim);
  }
  return transform_->convert(parent_->convert(restrictions, forbid_fake_dim), forbid_fake_dim);
}

Restrictions TransformStack::invert(const Restrictions& restrictions) const
{
  if (identity()) {
    return restrictions;
  }

  auto result = transform_->invert(restrictions);
  if (parent_->identity()) {
    return result;
  }
  // FIXME: This is very inefficient if the stack is large. Since each invert() takes by const
  // reference and returns a fresh Restructions (i.e. std::vector), we end up with a bunch of
  // temporary objects and allocations. We should be std::move-ing this!
  return parent_->invert(result);
}

tuple<std::uint64_t> TransformStack::invert_color(tuple<std::uint64_t> color) const
{
  if (identity()) {
    return color;
  }

  auto result = transform_->invert_color(std::move(color));
  if (parent_->identity()) {
    return result;
  }
  return parent_->invert_color(std::move(result));
}

tuple<std::uint64_t> TransformStack::invert_extents(const tuple<std::uint64_t>& extents) const
{
  if (identity()) {
    return extents;
  }

  auto result = transform_->invert_extents(extents);
  if (parent_->identity()) {
    return result;
  }
  // FIXME: This is very inefficient if the stack is large. Since each invert() takes by const
  // reference and returns a fresh Restructions (i.e. std::vector), we end up with a bunch of
  // temporary objects and allocations. We should be std::move-ing this!
  return parent_->invert_extents(result);
}

tuple<std::uint64_t> TransformStack::invert_point(const tuple<std::uint64_t>& point) const
{
  if (identity()) {
    return point;
  }

  auto result = transform_->invert_point(point);
  if (parent_->identity()) {
    return result;
  }
  // FIXME: This is very inefficient if the stack is large. Since each invert() takes by const
  // reference and returns a fresh Restructions (i.e. std::vector), we end up with a bunch of
  // temporary objects and allocations. We should be std::move-ing this!
  return parent_->invert_point(result);
}

void TransformStack::pack(BufferBuilder& buffer) const
{
  if (identity()) {
    buffer.pack<std::int32_t>(-1);
  } else {
    transform_->pack(buffer);
    parent_->pack(buffer);
  }
}

Legion::Domain TransformStack::transform(const Legion::Domain& input) const
{
  LegateAssert(transform_ != nullptr);
  return transform_->transform(parent_->identity() ? input : parent_->transform(input));
}

namespace {

Legion::DomainAffineTransform combine(const Legion::DomainAffineTransform& lhs,
                                      const Legion::DomainAffineTransform& rhs)
{
  Legion::DomainAffineTransform result;
  result.transform = lhs.transform * rhs.transform;
  result.offset    = lhs.transform * rhs.offset + lhs.offset;
  return result;
}

}  // namespace

Legion::DomainAffineTransform TransformStack::inverse_transform(std::int32_t in_dim) const
{
  LegateAssert(transform_ != nullptr);
  auto result  = transform_->inverse_transform(in_dim);
  auto out_dim = transform_->target_ndim(in_dim);

  if (parent_->identity()) {
    return result;
  }

  auto parent = parent_->inverse_transform(out_dim);
  return combine(parent, result);
}

void TransformStack::print(std::ostream& out) const
{
  if (identity()) {
    out << "(identity)";
    return;
  }

  transform_->print(out);
  if (!parent_->identity()) {
    out << " >> ";
    parent_->print(out);
  }
}

std::unique_ptr<StoreTransform> TransformStack::pop()
{
  LegateAssert(transform_ != nullptr);
  auto result = std::move(transform_);
  if (parent_) {
    transform_ = std::move(parent_->transform_);
    parent_    = std::move(parent_->parent_);
  }
  return result;
}

void TransformStack::dump() const
{
  // We are printing to cerr, we absolutely want the stream to be synchronized with the
  // underlying c lib streams
  std::cerr << *this << std::endl;  // NOLINT(performance-avoid-endl)
}

std::vector<std::int32_t> TransformStack::find_imaginary_dims() const
{
  std::vector<std::int32_t> dims;
  if (parent_) {
    dims = parent_->find_imaginary_dims();
  }
  if (transform_) {
    transform_->find_imaginary_dims(dims);
  }
  return dims;
}

Domain Shift::transform(const Domain& input) const
{
  auto result = input;
  result.rect_data[dim_] += offset_;
  result.rect_data[dim_ + result.dim] += offset_;
  return result;
}

Legion::DomainAffineTransform Shift::inverse_transform(std::int32_t in_dim) const
{
  LegateCheck(dim_ < in_dim);
  const auto out_dim = in_dim;
  Legion::DomainAffineTransform result;

  result.transform.m = out_dim;
  result.transform.n = in_dim;
  for (std::int32_t i = 0; i < out_dim; ++i) {
    for (std::int32_t j = 0; j < in_dim; ++j) {
      result.transform.matrix[i * in_dim + j] = static_cast<coord_t>(i == j);
    }
  }

  result.offset.dim = out_dim;
  for (std::int32_t i = 0; i < out_dim; ++i) {
    result.offset[i] = i == dim_ ? -offset_ : 0;
  }
  return result;
}

std::unique_ptr<Partition> Shift::convert(const Partition* partition) const
{
  switch (const auto kind = partition->kind()) {
    case Partition::Kind::NO_PARTITION: {
      return create_no_partition();
    }
    case Partition::Kind::TILING: {
      auto tiling = static_cast<const Tiling*>(partition);
      return create_tiling(tuple<std::uint64_t>{tiling->tile_shape()},
                           tuple<std::uint64_t>{tiling->color_shape()},
                           tiling->offsets().update(dim_, offset_));
    }
    default:
      throw std::invalid_argument{
        "Invalid partition kind: " +
        std::to_string(static_cast<std::underlying_type_t<decltype(kind)>>(kind))};
  }
  return {};
}

std::unique_ptr<Partition> Shift::invert(const Partition* partition) const
{
  switch (const auto kind = partition->kind()) {
    case Partition::Kind::NO_PARTITION: {
      return create_no_partition();
    }
    case Partition::Kind::TILING: {
      auto tiling     = static_cast<const Tiling*>(partition);
      auto new_offset = tiling->offsets()[dim_] - offset_;
      return create_tiling(tuple<std::uint64_t>{tiling->tile_shape()},
                           tuple<std::uint64_t>{tiling->color_shape()},
                           tiling->offsets().update(dim_, new_offset));
    }
    default:
      throw std::invalid_argument{
        "Invalid partition kind: " +
        std::to_string(static_cast<std::underlying_type_t<decltype(kind)>>(kind))};
  }
  return {};
}

tuple<std::uint64_t> Shift::invert_point(const tuple<std::uint64_t>& point) const
{
  auto result = point;
  result[dim_] -= offset_;
  return result;
}

void Shift::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(LEGATE_CORE_TRANSFORM_SHIFT);
  buffer.pack<std::int32_t>(dim_);
  buffer.pack<std::int64_t>(offset_);
}

void Shift::print(std::ostream& out) const
{
  out << "Shift(dim: " << dim_ << ", "
      << "offset: " << offset_ << ")";
}

Domain Promote::transform(const Domain& input) const
{
  Domain output;
  output.dim = input.dim + 1;

  for (std::int32_t out_dim = 0, in_dim = 0; out_dim < output.dim; ++out_dim) {
    if (out_dim == extra_dim_) {
      output.rect_data[out_dim]              = 0;
      output.rect_data[out_dim + output.dim] = dim_size_ - 1;
    } else {
      output.rect_data[out_dim]              = input.rect_data[in_dim];
      output.rect_data[out_dim + output.dim] = input.rect_data[in_dim + input.dim];
      ++in_dim;
    }
  }
  return output;
}

Legion::DomainAffineTransform Promote::inverse_transform(std::int32_t in_dim) const
{
  LegateCheck(extra_dim_ < in_dim);
  const auto out_dim = in_dim - 1;
  Legion::DomainAffineTransform result;

  result.transform.m = std::max<std::int32_t>(out_dim, 1);
  result.transform.n = in_dim;
  for (std::int32_t i = 0; i < result.transform.m; ++i) {
    for (std::int32_t j = 0; j < result.transform.n; ++j) {
      result.transform.matrix[i * in_dim + j] = 0;
    }
  }

  if (out_dim > 0) {
    for (std::int32_t j = 0, i = 0; j < result.transform.n; ++j) {
      if (j != extra_dim_) {
        result.transform.matrix[i++ * in_dim + j] = 1;
      }
    }
  }

  result.offset.dim = std::max<std::int32_t>(out_dim, 1);
  for (std::int32_t i = 0; i < result.transform.m; ++i) {
    result.offset[i] = 0;
  }

  return result;
}

std::unique_ptr<Partition> Promote::convert(const Partition* partition) const
{
  switch (const auto kind = partition->kind()) {
    case Partition::Kind::NO_PARTITION: {
      return create_no_partition();
    }
    case Partition::Kind::TILING: {
      auto tiling = static_cast<const Tiling*>(partition);
      return create_tiling(tiling->tile_shape().insert(extra_dim_, dim_size_),
                           tiling->color_shape().insert(extra_dim_, 1),
                           tiling->offsets().insert(extra_dim_, 0));
    }
    default:
      throw std::invalid_argument{
        "Invalid partition kind: " +
        std::to_string(static_cast<std::underlying_type_t<decltype(kind)>>(kind))};
  }
  return {};
}

std::unique_ptr<Partition> Promote::invert(const Partition* partition) const
{
  switch (const auto kind = partition->kind()) {
    case Partition::Kind::NO_PARTITION: {
      return create_no_partition();
    }
    case Partition::Kind::TILING: {
      auto tiling = static_cast<const Tiling*>(partition);
      return create_tiling(tiling->tile_shape().remove(extra_dim_),
                           tiling->color_shape().remove(extra_dim_),
                           tiling->offsets().remove(extra_dim_));
    }
    default:
      throw std::invalid_argument{
        "Invalid partition kind: " +
        std::to_string(static_cast<std::underlying_type_t<decltype(kind)>>(kind))};
  }
  return {};
}

proj::SymbolicPoint Promote::invert(const proj::SymbolicPoint& point) const
{
  return point.remove(extra_dim_);
}

Restrictions Promote::convert(const Restrictions& restrictions, bool forbid_fake_dim) const
{
  return restrictions.insert(extra_dim_,
                             forbid_fake_dim ? Restriction::FORBID : Restriction::AVOID);
}

Restrictions Promote::invert(const Restrictions& restrictions) const
{
  return restrictions.remove(extra_dim_);
}

tuple<std::uint64_t> Promote::invert_extents(const tuple<std::uint64_t>& extents) const
{
  return extents.remove(extra_dim_);
}

tuple<std::uint64_t> Promote::invert_point(const tuple<std::uint64_t>& point) const
{
  return point.remove(extra_dim_);
}

void Promote::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(LEGATE_CORE_TRANSFORM_PROMOTE);
  buffer.pack<std::int32_t>(extra_dim_);
  buffer.pack<std::int64_t>(dim_size_);
}

void Promote::print(std::ostream& out) const
{
  out << "Promote(";
  out << "extra_dim: " << extra_dim_ << ", ";
  out << "dim_size: " << dim_size_ << ")";
}

void Promote::find_imaginary_dims(std::vector<std::int32_t>& dims) const
{
  for (auto& dim : dims) {
    if (dim >= extra_dim_) {
      dim++;
    }
  }
  dims.push_back(extra_dim_);
}

Domain Project::transform(const Domain& input) const
{
  Domain output;
  output.dim = input.dim - 1;

  for (std::int32_t in_dim = 0, out_dim = 0; in_dim < input.dim; ++in_dim) {
    if (in_dim != dim_) {
      output.rect_data[out_dim]              = input.rect_data[in_dim];
      output.rect_data[out_dim + output.dim] = input.rect_data[in_dim + input.dim];
      ++out_dim;
    }
  }
  return output;
}

Legion::DomainAffineTransform Project::inverse_transform(std::int32_t in_dim) const
{
  auto out_dim = in_dim + 1;
  LegateCheck(dim_ < out_dim);
  Legion::DomainAffineTransform result;

  result.transform.m = out_dim;
  if (in_dim == 0) {
    result.transform.n         = out_dim;
    result.transform.matrix[0] = 0;
  } else {
    result.transform.n = in_dim;
    for (std::int32_t i = 0; i < out_dim; ++i) {
      for (std::int32_t j = 0; j < in_dim; ++j) {
        result.transform.matrix[i * in_dim + j] = 0;
      }
    }

    for (std::int32_t i = 0, j = 0; i < out_dim; ++i) {
      if (i != dim_) {
        result.transform.matrix[i * in_dim + j++] = 1;
      }
    }
  }

  result.offset.dim = out_dim;
  for (std::int32_t i = 0; i < out_dim; ++i) {
    result.offset[i] = i == dim_ ? coord_ : 0;
  }

  return result;
}

std::unique_ptr<Partition> Project::convert(const Partition* partition) const
{
  switch (const auto kind = partition->kind()) {
    case Partition::Kind::NO_PARTITION: {
      return create_no_partition();
    }
    case Partition::Kind::TILING: {
      auto tiling = static_cast<const Tiling*>(partition);
      return create_tiling(tiling->tile_shape().remove(dim_),
                           tiling->color_shape().remove(dim_),
                           tiling->offsets().remove(dim_));
    }
    default:
      throw std::invalid_argument{
        "Invalid partition kind: " +
        std::to_string(static_cast<std::underlying_type_t<decltype(kind)>>(kind))};
  }
  return {};
}

std::unique_ptr<Partition> Project::invert(const Partition* partition) const
{
  switch (const auto kind = partition->kind()) {
    case Partition::Kind::NO_PARTITION: {
      return create_no_partition();
    }
    case Partition::Kind::TILING: {
      auto tiling = static_cast<const Tiling*>(partition);
      return create_tiling(tiling->tile_shape().insert(dim_, 1),
                           tiling->color_shape().insert(dim_, 1),
                           tiling->offsets().insert(dim_, coord_));
    }
    default:
      throw std::invalid_argument{
        "Invalid partition kind: " +
        std::to_string(static_cast<std::underlying_type_t<decltype(kind)>>(kind))};
  }
  return {};
}

proj::SymbolicPoint Project::invert(const proj::SymbolicPoint& point) const
{
  return point.insert(dim_, proj::SymbolicExpr());
}

Restrictions Project::convert(const Restrictions& restrictions, bool /*forbid_fake_dim*/) const
{
  return restrictions.remove(dim_);
}

Restrictions Project::invert(const Restrictions& restrictions) const
{
  return restrictions.insert(dim_, Restriction::ALLOW);
}

tuple<std::uint64_t> Project::invert_color(tuple<std::uint64_t> color) const
{
  return color.insert(dim_, 0);
}

tuple<std::uint64_t> Project::invert_extents(const tuple<std::uint64_t>& extents) const
{
  return extents.insert(dim_, 1);
}

tuple<std::uint64_t> Project::invert_point(const tuple<std::uint64_t>& point) const
{
  return point.insert(dim_, coord_);
}

void Project::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(LEGATE_CORE_TRANSFORM_PROJECT);
  buffer.pack<std::int32_t>(dim_);
  buffer.pack<std::int64_t>(coord_);
}

void Project::print(std::ostream& out) const
{
  out << "Project(";
  out << "dim: " << dim_ << ", ";
  out << "coord: " << coord_ << ")";
}

void Project::find_imaginary_dims(std::vector<std::int32_t>& dims) const
{
  auto finder = std::find(dims.begin(), dims.end(), dim_);
  if (finder != dims.end()) {
    dims.erase(finder);
  }
  for (auto& dim : dims) {
    if (dim > dim_) {
      --dim;
    }
  }
}

Transpose::Transpose(std::vector<std::int32_t>&& axes) : axes_{std::move(axes)}
{
  const auto size = axes_.size();
  // could alternatively do
  //
  // inverse_.resize(size);
  // std::iota(inverse_.begin(), inverse_.end(), 0);
  //
  // but this results in 2 traversals of the array, once to initialize inverse_ to 0, and a
  // second time to do the iota-ing
  inverse_.reserve(size);
  std::generate_n(std::back_inserter(inverse_), size, [n = 0]() mutable { return n++; });
  std::sort(inverse_.begin(), inverse_.end(), [&](const int32_t& idx1, const int32_t& idx2) {
    return axes_[idx1] < axes_[idx2];
  });
}

Domain Transpose::transform(const Domain& domain) const
{
  Domain output;
  output.dim = domain.dim;
  for (std::int32_t out_dim = 0; out_dim < output.dim; ++out_dim) {
    auto in_dim                            = axes_[out_dim];
    output.rect_data[out_dim]              = domain.rect_data[in_dim];
    output.rect_data[out_dim + output.dim] = domain.rect_data[in_dim + domain.dim];
  }
  return output;
}

Legion::DomainAffineTransform Transpose::inverse_transform(std::int32_t in_dim) const
{
  Legion::DomainAffineTransform result;

  result.transform.m = in_dim;
  result.transform.n = in_dim;
  for (std::int32_t i = 0; i < in_dim; ++i) {
    for (std::int32_t j = 0; j < in_dim; ++j) {
      result.transform.matrix[i * in_dim + j] = 0;
    }
  }

  for (std::int32_t j = 0; j < in_dim; ++j) {
    result.transform.matrix[axes_[j] * in_dim + j] = 1;
  }

  result.offset.dim = in_dim;
  for (std::int32_t i = 0; i < in_dim; ++i) {
    result.offset[i] = 0;
  }

  return result;
}

std::unique_ptr<Partition> Transpose::convert(const Partition* partition) const
{
  switch (const auto kind = partition->kind()) {
    case Partition::Kind::NO_PARTITION: {
      return create_no_partition();
    }
    case Partition::Kind::TILING: {
      auto tiling = static_cast<const Tiling*>(partition);
      return create_tiling(tiling->tile_shape().map(axes_),
                           tiling->color_shape().map(axes_),
                           tiling->offsets().map(axes_));
    }
    default:
      throw std::invalid_argument{
        "Invalid partition kind: " +
        std::to_string(static_cast<std::underlying_type_t<decltype(kind)>>(kind))};
  }
  return {};
}

std::unique_ptr<Partition> Transpose::invert(const Partition* partition) const
{
  switch (const auto kind = partition->kind()) {
    case Partition::Kind::NO_PARTITION: {
      return create_no_partition();
    }
    case Partition::Kind::TILING: {
      auto tiling = static_cast<const Tiling*>(partition);
      return create_tiling(tiling->tile_shape().map(inverse_),
                           tiling->color_shape().map(inverse_),
                           tiling->offsets().map(inverse_));
    }
    default:
      throw std::invalid_argument{
        "Invalid partition kind: " +
        std::to_string(static_cast<std::underlying_type_t<decltype(kind)>>(kind))};
  }
  return {};
}

proj::SymbolicPoint Transpose::invert(const proj::SymbolicPoint& point) const
{
  std::vector<proj::SymbolicExpr> exprs;

  exprs.reserve(inverse_.size());
  for (auto&& idx : inverse_) {
    exprs.emplace_back(point[idx]);
  }
  return proj::SymbolicPoint{std::move(exprs)};
}

Restrictions Transpose::convert(const Restrictions& restrictions, bool /*forbid_fake_dim*/) const
{
  std::vector<Restriction> result;

  result.reserve(axes_.size());
  for (auto dim : axes_) {
    result.emplace_back(restrictions[dim]);
  }
  return Restrictions{std::move(result)};
}

Restrictions Transpose::invert(const Restrictions& restrictions) const
{
  std::vector<Restriction> result;

  result.reserve(inverse_.size());
  for (auto dim : inverse_) {
    result.emplace_back(restrictions[dim]);
  }
  return Restrictions{std::move(result)};
}

tuple<std::uint64_t> Transpose::invert_extents(const tuple<std::uint64_t>& extents) const
{
  return extents.map(inverse_);
}

tuple<std::uint64_t> Transpose::invert_point(const tuple<std::uint64_t>& point) const
{
  return point.map(inverse_);
}

namespace {  // anonymous

template <typename T>
void print_vector(std::ostream& out, const std::vector<T>& vec)
{
  bool past_first = false;
  out << "[";
  for (const T& val : vec) {
    if (past_first) {
      out << ", ";
    } else {
      past_first = true;
    }
    out << val;
  }
  out << "]";
}

}  // anonymous namespace

void Transpose::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(LEGATE_CORE_TRANSFORM_TRANSPOSE);
  buffer.pack<std::uint32_t>(axes_.size());
  for (auto axis : axes_) {
    buffer.pack<std::int32_t>(axis);
  }
}

void Transpose::print(std::ostream& out) const
{
  out << "Transpose(";
  out << "axes: ";
  print_vector(out, axes_);
  out << ")";
}

void Transpose::find_imaginary_dims(std::vector<std::int32_t>& dims) const
{
  // i should be added to X.tranpose(axes).promoted iff axes[i] is in X.promoted
  // e.g. X.promoted = [0] => X.transpose((1,2,0)).promoted = [2]
  for (auto& promoted : dims) {
    auto finder = std::find(axes_.begin(), axes_.end(), promoted);

    LegateCheck(finder != axes_.end());
    promoted = static_cast<std::int32_t>(finder - axes_.begin());
  }
}

Delinearize::Delinearize(std::int32_t dim, std::vector<std::uint64_t>&& sizes)
  : dim_{dim}, sizes_{std::move(sizes)}, strides_(sizes_.size(), 1), volume_{1}
{
  for (auto size_dim = static_cast<std::int32_t>(sizes_.size() - 2); size_dim >= 0; --size_dim) {
    strides_[size_dim] = strides_[size_dim + 1] * sizes_[size_dim + 1];
  }
  for (auto size : sizes_) {
    volume_ *= size;
  }
}

Domain Delinearize::transform(const Domain& domain) const
{
  auto delinearize = [&](const auto dim, const auto ndim, const auto& strides) {
    Domain output;
    output.dim = domain.dim - 1 + ndim;
    for (std::int32_t in_dim = 0, out_dim = 0; in_dim < domain.dim; ++in_dim) {
      if (in_dim == dim) {
        auto lo = domain.rect_data[in_dim];
        auto hi = domain.rect_data[domain.dim + in_dim];
        for (auto stride : strides) {
          output.rect_data[out_dim]              = lo / stride;
          output.rect_data[output.dim + out_dim] = hi / stride;
          lo                                     = lo % stride;
          hi                                     = hi % stride;
          ++out_dim;
        }
      } else {
        output.rect_data[out_dim]              = domain.rect_data[in_dim];
        output.rect_data[output.dim + out_dim] = domain.rect_data[domain.dim + in_dim];
        ++out_dim;
      }
    }
    return output;
  };
  return delinearize(dim_, sizes_.size(), strides_);
}

Legion::DomainAffineTransform Delinearize::inverse_transform(std::int32_t in_dim) const
{
  const auto out_dim = static_cast<std::int32_t>(in_dim - strides_.size() + 1);
  Legion::DomainAffineTransform result;

  result.transform.m = out_dim;
  result.transform.n = in_dim;
  for (std::int32_t i = 0; i < out_dim; ++i) {
    for (std::int32_t j = 0; j < in_dim; ++j) {
      result.transform.matrix[i * in_dim + j] = 0;
    }
  }

  for (std::int32_t i = 0, j = 0; i < out_dim; ++i) {
    if (i == dim_) {
      for (auto stride : strides_) {
        result.transform.matrix[i * in_dim + j++] = static_cast<coord_t>(stride);
      }
    } else {
      result.transform.matrix[i * in_dim + j++] = 1;
    }
  }

  result.offset.dim = out_dim;
  for (std::int32_t i = 0; i < out_dim; ++i) {
    result.offset[i] = 0;
  }

  return result;
}

std::unique_ptr<Partition> Delinearize::convert(const Partition* /*partition*/) const
{
  throw NonInvertibleTransformation{"Delinearize transform cannot be used in conversion"};
  return {};
}

std::unique_ptr<Partition> Delinearize::invert(const Partition* partition) const
{
  switch (const auto kind = partition->kind()) {
    case Partition::Kind::NO_PARTITION: {
      return create_no_partition();
    }
    case Partition::Kind::TILING: {
      const auto tiling = static_cast<const Tiling*>(partition);
      auto& tile_shape  = tiling->tile_shape();
      auto& color_shape = tiling->color_shape();
      auto& offsets     = tiling->offsets();

      const auto invertible = [&] {
        std::size_t volume     = 1;
        std::size_t sum_offset = 0;
        for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
          volume *= color_shape[dim_ + idx];
          sum_offset += offsets[dim_ + idx];
        }
        return 1 == volume && 0 == sum_offset;
      };

      if (!invertible()) {
        throw NonInvertibleTransformation{"Delinearize transform cannot invert this partition: " +
                                          tiling->to_string()};
      }

      auto new_tile_shape  = tile_shape;
      auto new_color_shape = color_shape;
      auto new_offsets     = offsets;

      for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
        new_tile_shape.remove_inplace(dim_ + 1);
        new_color_shape.remove_inplace(dim_ + 1);
        new_offsets.remove_inplace(dim_ + 1);
      }

      new_tile_shape[dim_] *= strides_[0];
      new_offsets[dim_] *= static_cast<std::int64_t>(strides_[0]);

      return create_tiling(
        std::move(new_tile_shape), std::move(new_color_shape), std::move(new_offsets));
    }
    default:
      throw std::invalid_argument{
        "Invalid partition kind: " +
        std::to_string(static_cast<std::underlying_type_t<decltype(kind)>>(kind))};
  }
  return {};
}

proj::SymbolicPoint Delinearize::invert(const proj::SymbolicPoint& point) const
{
  std::vector<proj::SymbolicExpr> exprs;

  exprs.reserve(point.size() - (sizes_.size() - 1));
  for (std::int32_t dim = 0; dim < dim_ + 1; ++dim) {
    exprs.push_back(point[dim]);
  }
  for (auto dim = dim_ + sizes_.size(); dim < point.size(); ++dim) {
    exprs.push_back(point[dim]);
  }
  return proj::SymbolicPoint{std::move(exprs)};
}

Restrictions Delinearize::convert(const Restrictions& restrictions, bool /*forbid_fake_dim*/) const
{
  std::vector<Restriction> result;

  result.reserve(restrictions.size() + (sizes_.size() - 1));
  for (auto dim = 0; dim <= dim_; ++dim) {
    result.emplace_back(restrictions[dim]);
  }
  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    result.emplace_back(Restriction::FORBID);
  }
  for (std::uint32_t dim = dim_ + 1; dim < restrictions.size(); ++dim) {
    result.emplace_back(restrictions[dim]);
  }
  return Restrictions{std::move(result)};
}

Restrictions Delinearize::invert(const Restrictions& restrictions) const
{
  std::vector<Restriction> result;

  result.reserve(restrictions.size() - (sizes_.size() - 1));
  for (auto dim = 0; dim <= dim_; ++dim) {
    result.emplace_back(restrictions[dim]);
  }
  for (auto dim = dim_ + sizes_.size(); dim < restrictions.size(); ++dim) {
    result.emplace_back(restrictions[dim]);
  }
  return Restrictions{std::move(result)};
}

tuple<std::uint64_t> Delinearize::invert_color(tuple<std::uint64_t> /*color*/) const
{
  throw NonInvertibleTransformation{};
  return {};
}

tuple<std::uint64_t> Delinearize::invert_extents(const tuple<std::uint64_t>& /*extents*/) const
{
  throw NonInvertibleTransformation{};
  return {};
}

tuple<std::uint64_t> Delinearize::invert_point(const tuple<std::uint64_t>& /*point*/) const
{
  throw NonInvertibleTransformation{};
  return {};
}

void Delinearize::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(LEGATE_CORE_TRANSFORM_DELINEARIZE);
  buffer.pack<std::int32_t>(dim_);
  buffer.pack<std::uint32_t>(sizes_.size());
  for (auto extent : sizes_) {
    buffer.pack<std::uint64_t>(extent);
  }
}

void Delinearize::print(std::ostream& out) const
{
  out << "Delinearize(";
  out << "dim: " << dim_ << ", ";
  out << "sizes: ";
  print_vector(out, sizes_);
  out << ")";
}

std::int32_t Delinearize::target_ndim(std::int32_t source_ndim) const
{
  return static_cast<std::int32_t>(source_ndim - strides_.size() + 1);
}

std::ostream& operator<<(std::ostream& out, const Transform& transform)
{
  transform.print(out);
  return out;
}

}  // namespace legate::detail
