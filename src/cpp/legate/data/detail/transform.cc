/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform.h>

#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace legate::detail {

Restrictions TransformStack::convert(Restrictions restrictions, bool forbid_fake_dim) const
{
  return convert_(
    [&](auto&& transform, auto&& input) {
      return transform->convert(std::forward<decltype(input)>(input), forbid_fake_dim);
    },
    std::move(restrictions));
}

tuple<std::uint64_t> TransformStack::convert_color(tuple<std::uint64_t> color) const
{
  return convert_(
    [&](auto&& transform, auto&& input) {
      return transform->convert_color(std::forward<decltype(input)>(input));
    },
    std::move(color));
}

tuple<std::uint64_t> TransformStack::convert_color_shape(tuple<std::uint64_t> color_shape) const
{
  return convert_(
    [&](auto&& transform, auto&& input) {
      return transform->convert_color_shape(std::forward<decltype(input)>(input));
    },
    std::move(color_shape));
}

tuple<std::int64_t> TransformStack::convert_point(tuple<std::int64_t> point) const
{
  return convert_(
    [&](auto&& transform, auto&& input) {
      return transform->convert_point(std::forward<decltype(input)>(input));
    },
    std::move(point));
}

tuple<std::uint64_t> TransformStack::convert_extents(tuple<std::uint64_t> extents) const
{
  return convert_(
    [&](auto&& transform, auto&& input) {
      return transform->convert_extents(std::forward<decltype(input)>(input));
    },
    std::move(extents));
}

proj::SymbolicPoint TransformStack::invert(proj::SymbolicPoint point) const
{
  return invert_(
    [&](auto&& transform, auto&& input) {
      return transform->invert(std::forward<decltype(input)>(input));
    },
    std::move(point));
}

Restrictions TransformStack::invert(Restrictions restrictions) const
{
  return invert_(
    [&](auto&& transform, auto&& input) {
      return transform->invert(std::forward<decltype(input)>(input));
    },
    std::move(restrictions));
}

tuple<std::uint64_t> TransformStack::invert_color(tuple<std::uint64_t> color) const
{
  return invert_(
    [&](auto&& transform, auto&& input) {
      return transform->invert_color(std::forward<decltype(input)>(input));
    },
    std::move(color));
}

tuple<std::uint64_t> TransformStack::invert_color_shape(tuple<std::uint64_t> color_shape) const
{
  return invert_(
    [&](auto&& transform, auto&& input) {
      return transform->invert_color_shape(std::forward<decltype(input)>(input));
    },
    std::move(color_shape));
}

tuple<std::int64_t> TransformStack::invert_point(tuple<std::int64_t> point) const
{
  return invert_(
    [&](auto&& transform, auto&& input) {
      return transform->invert_point(std::forward<decltype(input)>(input));
    },
    std::move(point));
}

tuple<std::uint64_t> TransformStack::invert_extents(tuple<std::uint64_t> extents) const
{
  return invert_(
    [&](auto&& transform, auto&& input) {
      return transform->invert_extents(std::forward<decltype(input)>(input));
    },
    std::move(extents));
}

void TransformStack::pack(BufferBuilder& buffer) const
{
  if (identity()) {
    buffer.pack<CoreTransform>(CoreTransform::INVALID);
  } else {
    transform_->pack(buffer);
    parent_->pack(buffer);
  }
}

Legion::Domain TransformStack::transform(const Legion::Domain& input) const
{
  LEGATE_ASSERT(transform_ != nullptr);
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
  LEGATE_ASSERT(transform_ != nullptr);
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
  LEGATE_ASSERT(transform_ != nullptr);
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

// ==========================================================================================

Domain Shift::transform(const Domain& input) const
{
  auto result = input;
  result.rect_data[dim_] += offset_;
  result.rect_data[dim_ + result.dim] += offset_;
  return result;
}

Legion::DomainAffineTransform Shift::inverse_transform(std::int32_t in_dim) const
{
  LEGATE_CHECK(dim_ < in_dim);
  const auto out_dim = in_dim;
  Legion::DomainAffineTransform result;

  result.transform.m = out_dim;
  result.transform.n = in_dim;
  for (std::int32_t i = 0; i < out_dim; ++i) {
    for (std::int32_t j = 0; j < in_dim; ++j) {
      result.transform.matrix[(i * in_dim) + j] = static_cast<coord_t>(i == j);
    }
  }

  result.offset.dim = out_dim;
  for (std::int32_t i = 0; i < out_dim; ++i) {
    result.offset[i] = i == dim_ ? -offset_ : 0;
  }
  return result;
}

tuple<std::int64_t> Shift::convert_point(tuple<std::int64_t> point) const
{
  point[dim_] += offset_;
  return point;
}

tuple<std::int64_t> Shift::invert_point(tuple<std::int64_t> point) const
{
  point[dim_] -= offset_;
  return point;
}

void Shift::pack(BufferBuilder& buffer) const
{
  buffer.pack<CoreTransform>(CoreTransform::SHIFT);
  buffer.pack<std::int32_t>(dim_);
  buffer.pack<std::int64_t>(offset_);
}

void Shift::print(std::ostream& out) const
{
  out << "Shift(dim: " << dim_ << ", "
      << "offset: " << offset_ << ")";
}

// ==========================================================================================

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
  LEGATE_CHECK(extra_dim_ < in_dim);
  const auto out_dim = in_dim - 1;
  Legion::DomainAffineTransform result;

  result.transform.m = std::max<std::int32_t>(out_dim, 1);
  result.transform.n = in_dim;
  for (std::int32_t i = 0; i < result.transform.m; ++i) {
    for (std::int32_t j = 0; j < result.transform.n; ++j) {
      result.transform.matrix[(i * in_dim) + j] = 0;
    }
  }

  if (out_dim > 0) {
    for (std::int32_t j = 0, i = 0; j < result.transform.n; ++j) {
      if (j != extra_dim_) {
        result.transform.matrix[(i++ * in_dim) + j] = 1;
      }
    }
  }

  result.offset.dim = std::max<std::int32_t>(out_dim, 1);
  for (std::int32_t i = 0; i < result.transform.m; ++i) {
    result.offset[i] = 0;
  }

  return result;
}

Restrictions Promote::convert(Restrictions restrictions, bool forbid_fake_dim) const
{
  restrictions.insert_inplace(extra_dim_,
                              forbid_fake_dim ? Restriction::FORBID : Restriction::AVOID);
  return restrictions;
}

tuple<std::uint64_t> Promote::convert_color(tuple<std::uint64_t> color) const
{
  color.insert_inplace(extra_dim_, 0);
  return color;
}

tuple<std::uint64_t> Promote::convert_color_shape(tuple<std::uint64_t> color_shape) const
{
  color_shape.insert_inplace(extra_dim_, 1);
  return color_shape;
}

tuple<std::int64_t> Promote::convert_point(tuple<std::int64_t> point) const
{
  point.insert_inplace(extra_dim_, 0);
  return point;
}

tuple<std::uint64_t> Promote::convert_extents(tuple<std::uint64_t> extents) const
{
  extents.insert_inplace(extra_dim_, dim_size_);
  return extents;
}

proj::SymbolicPoint Promote::invert(proj::SymbolicPoint point) const
{
  point.remove_inplace(extra_dim_);
  return point;
}

Restrictions Promote::invert(Restrictions restrictions) const
{
  restrictions.remove_inplace(extra_dim_);
  return restrictions;
}

tuple<std::uint64_t> Promote::invert_color(tuple<std::uint64_t> color) const
{
  color.remove_inplace(extra_dim_);
  return color;
}

tuple<std::uint64_t> Promote::invert_color_shape(tuple<std::uint64_t> color_shape) const
{
  color_shape.remove_inplace(extra_dim_);
  return color_shape;
}

tuple<std::int64_t> Promote::invert_point(tuple<std::int64_t> point) const
{
  point.remove_inplace(extra_dim_);
  return point;
}

tuple<std::uint64_t> Promote::invert_extents(tuple<std::uint64_t> extents) const
{
  extents.remove_inplace(extra_dim_);
  return extents;
}

void Promote::pack(BufferBuilder& buffer) const
{
  buffer.pack<CoreTransform>(CoreTransform::PROMOTE);
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
  for (auto&& dim : dims) {
    if (dim >= extra_dim_) {
      dim++;
    }
  }
  dims.push_back(extra_dim_);
}

// ==========================================================================================

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
  LEGATE_CHECK(dim_ < out_dim);
  Legion::DomainAffineTransform result;

  result.transform.m = out_dim;
  if (in_dim == 0) {
    result.transform.n         = out_dim;
    result.transform.matrix[0] = 0;
  } else {
    result.transform.n = in_dim;
    for (std::int32_t i = 0; i < out_dim; ++i) {
      for (std::int32_t j = 0; j < in_dim; ++j) {
        result.transform.matrix[(i * in_dim) + j] = 0;
      }
    }

    for (std::int32_t i = 0, j = 0; i < out_dim; ++i) {
      if (i != dim_) {
        result.transform.matrix[(i * in_dim) + j++] = 1;
      }
    }
  }

  result.offset.dim = out_dim;
  for (std::int32_t i = 0; i < out_dim; ++i) {
    result.offset[i] = i == dim_ ? coord_ : 0;
  }

  return result;
}

Restrictions Project::convert(Restrictions restrictions, bool /*forbid_fake_dim*/) const
{
  restrictions.remove_inplace(dim_);
  return restrictions;
}

tuple<std::uint64_t> Project::convert_color(tuple<std::uint64_t> color) const
{
  color.remove_inplace(dim_);
  return color;
}

tuple<std::uint64_t> Project::convert_color_shape(tuple<std::uint64_t> color_shape) const
{
  color_shape.remove_inplace(dim_);
  return color_shape;
}

tuple<std::int64_t> Project::convert_point(tuple<std::int64_t> point) const
{
  point.remove_inplace(dim_);
  return point;
}

tuple<std::uint64_t> Project::convert_extents(tuple<std::uint64_t> extents) const
{
  extents.remove_inplace(dim_);
  return extents;
}

proj::SymbolicPoint Project::invert(proj::SymbolicPoint point) const
{
  point.insert_inplace(dim_, proj::SymbolicExpr{});
  return point;
}

Restrictions Project::invert(Restrictions restrictions) const
{
  restrictions.insert_inplace(dim_, Restriction::ALLOW);
  return restrictions;
}

tuple<std::uint64_t> Project::invert_color(tuple<std::uint64_t> color) const
{
  color.insert_inplace(dim_, 0);
  return color;
}

tuple<std::uint64_t> Project::invert_color_shape(tuple<std::uint64_t> color_shape) const
{
  color_shape.insert_inplace(dim_, 1);
  return color_shape;
}

tuple<std::int64_t> Project::invert_point(tuple<std::int64_t> point) const
{
  point.insert_inplace(dim_, coord_);
  return point;
}

tuple<std::uint64_t> Project::invert_extents(tuple<std::uint64_t> extents) const
{
  extents.insert_inplace(dim_, 1);
  return extents;
}

void Project::pack(BufferBuilder& buffer) const
{
  buffer.pack<CoreTransform>(CoreTransform::PROJECT);
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
  for (auto&& dim : dims) {
    if (dim > dim_) {
      --dim;
    }
  }
}

// ==========================================================================================

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
  std::sort(inverse_.begin(), inverse_.end(), [&](std::int32_t idx1, std::int32_t idx2) {
    return axes_[static_cast<std::size_t>(idx1)] < axes_[static_cast<std::size_t>(idx2)];
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
      result.transform.matrix[(i * in_dim) + j] = 0;
    }
  }

  for (std::int32_t j = 0; j < in_dim; ++j) {
    result.transform.matrix[(axes_[j] * in_dim) + j] = 1;
  }

  result.offset.dim = in_dim;
  for (std::int32_t i = 0; i < in_dim; ++i) {
    result.offset[i] = 0;
  }

  return result;
}

Restrictions Transpose::convert(Restrictions restrictions, bool /*forbid_fake_dim*/) const
{
  // No in-place available
  return restrictions.map(axes_);
}

tuple<std::uint64_t> Transpose::convert_color(tuple<std::uint64_t> color) const
{
  // No in-place available
  return color.map(axes_);
}

tuple<std::uint64_t> Transpose::convert_color_shape(tuple<std::uint64_t> color_shape) const
{
  // No in-place available
  return color_shape.map(axes_);
}

tuple<std::int64_t> Transpose::convert_point(tuple<std::int64_t> point) const
{
  // No in-place available
  return point.map(axes_);
}

tuple<std::uint64_t> Transpose::convert_extents(tuple<std::uint64_t> extents) const
{
  // No in-place available
  return extents.map(axes_);
}

proj::SymbolicPoint Transpose::invert(proj::SymbolicPoint point) const
{
  // No in-place available
  return point.map(inverse_);
}

Restrictions Transpose::invert(Restrictions restrictions) const
{
  // No in-place available
  return restrictions.map(inverse_);
}

tuple<std::uint64_t> Transpose::invert_color(tuple<std::uint64_t> color) const
{
  // No in-place available
  return color.map(inverse_);
}

tuple<std::uint64_t> Transpose::invert_color_shape(tuple<std::uint64_t> color_shape) const
{
  // No in-place available
  return color_shape.map(inverse_);
}

tuple<std::int64_t> Transpose::invert_point(tuple<std::int64_t> point) const
{
  // No in-place available
  return point.map(inverse_);
}

tuple<std::uint64_t> Transpose::invert_extents(tuple<std::uint64_t> extents) const
{
  // No in-place available
  return extents.map(inverse_);
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
  buffer.pack<CoreTransform>(CoreTransform::TRANSPOSE);
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
  for (auto&& promoted : dims) {
    auto finder = std::find(axes_.begin(), axes_.end(), promoted);

    LEGATE_CHECK(finder != axes_.end());
    promoted = static_cast<std::int32_t>(finder - axes_.begin());
  }
}

// ==========================================================================================

Delinearize::Delinearize(std::int32_t dim, std::vector<std::uint64_t>&& sizes)
  : dim_{dim}, sizes_{std::move(sizes)}, strides_(sizes_.size(), 1), volume_{1}
{
  // Need this double cast since sizes_.size() might be < 2, and since the condition is >= 0,
  // we cannot just use std::size_t here
  for (auto size_dim = static_cast<std::int32_t>(sizes_.size() - 2); size_dim >= 0; --size_dim) {
    const auto usize_dim = static_cast<std::size_t>(size_dim);

    strides_[usize_dim] = strides_[usize_dim + 1] * sizes_[usize_dim + 1];
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
      result.transform.matrix[(i * in_dim) + j] = 0;
    }
  }

  for (std::int32_t i = 0, j = 0; i < out_dim; ++i) {
    if (i == dim_) {
      for (auto stride : strides_) {
        result.transform.matrix[(i * in_dim) + j++] = static_cast<coord_t>(stride);
      }
    } else {
      result.transform.matrix[(i * in_dim) + j++] = 1;
    }
  }

  result.offset.dim = out_dim;
  for (std::int32_t i = 0; i < out_dim; ++i) {
    result.offset[i] = 0;
  }

  return result;
}

Restrictions Delinearize::convert(Restrictions restrictions, bool /*forbid_fake_dim*/) const
{
  Restrictions result;

  result.reserve(restrictions.size() + (sizes_.size() - 1));
  for (auto dim = 0; dim <= dim_; ++dim) {
    result.append_inplace(restrictions[dim]);
  }
  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    result.append_inplace(Restriction::FORBID);
  }
  for (std::uint32_t dim = dim_ + 1; dim < restrictions.size(); ++dim) {
    result.append_inplace(restrictions[dim]);
  }
  return result;
}

tuple<std::uint64_t> Delinearize::convert_color(tuple<std::uint64_t> /*color*/) const
{
  throw TracedException<NonInvertibleTransformation>{};
  return {};
}

tuple<std::uint64_t> Delinearize::convert_color_shape(tuple<std::uint64_t> /*color_shape*/) const
{
  throw TracedException<NonInvertibleTransformation>{};
  return {};
}

tuple<std::uint64_t> Delinearize::convert_extents(tuple<std::uint64_t> /*extents*/) const
{
  throw TracedException<NonInvertibleTransformation>{};
  return {};
}

tuple<std::int64_t> Delinearize::convert_point(tuple<std::int64_t> /*point*/) const
{
  throw TracedException<NonInvertibleTransformation>{};
  return {};
}

proj::SymbolicPoint Delinearize::invert(proj::SymbolicPoint point) const
{
  proj::SymbolicPoint exprs;

  exprs.reserve(point.size() - (sizes_.size() - 1));
  for (std::int32_t dim = 0; dim < dim_ + 1; ++dim) {
    exprs.append_inplace(point[dim]);
  }
  for (auto dim = dim_ + sizes_.size(); dim < point.size(); ++dim) {
    exprs.append_inplace(point[dim]);
  }
  return exprs;
}

Restrictions Delinearize::invert(Restrictions restrictions) const
{
  Restrictions result;

  result.reserve(restrictions.size() - (sizes_.size() - 1));
  for (auto dim = 0; dim <= dim_; ++dim) {
    result.append_inplace(restrictions[dim]);
  }

  for (auto dim = dim_ + sizes_.size(); dim < restrictions.size(); ++dim) {
    result.append_inplace(restrictions[dim]);
  }
  return result;
}

tuple<std::uint64_t> Delinearize::invert_color(tuple<std::uint64_t> color) const
{
  auto sum = std::uint64_t{0};
  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    sum += color[dim_ + idx];
  }

  if (sum != 0) {
    throw TracedException<NonInvertibleTransformation>{};
  }

  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    color.remove_inplace(dim_ + 1);
  }

  return color;
}

tuple<std::uint64_t> Delinearize::invert_color_shape(tuple<std::uint64_t> color_shape) const
{
  auto volume = std::uint64_t{1};
  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    volume *= color_shape[dim_ + idx];
  }

  if (volume != 1) {
    throw TracedException<NonInvertibleTransformation>{};
  }

  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    color_shape.remove_inplace(dim_ + 1);
  }
  return color_shape;
}

tuple<std::int64_t> Delinearize::invert_point(tuple<std::int64_t> point) const
{
  auto sum = std::uint64_t{0};
  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    sum += point[dim_ + idx];
  }

  if (sum != 0) {
    throw TracedException<NonInvertibleTransformation>{};
  }

  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    point.remove_inplace(dim_ + 1);
  }
  point[dim_] *= static_cast<std::int64_t>(strides_[0]);

  return point;
}

tuple<std::uint64_t> Delinearize::invert_extents(tuple<std::uint64_t> extents) const
{
  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    if (extents[dim_ + idx] != sizes_[idx]) {
      throw TracedException<NonInvertibleTransformation>{};
    }
  }

  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    extents.remove_inplace(dim_ + 1);
  }
  extents[dim_] *= strides_[0];

  return extents;
}

void Delinearize::pack(BufferBuilder& buffer) const
{
  buffer.pack<CoreTransform>(CoreTransform::DELINEARIZE);
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
