/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform.h>

#include <legate/utilities/detail/array_algorithms.h>
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
      return transform->convert(std::forward<std::decay_t<decltype(input)>>(input),
                                forbid_fake_dim);
    },
    std::move(restrictions));
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> TransformStack::convert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  return convert_(
    [&](auto&& transform, auto&& input) {
      return transform->convert_color(std::forward<std::decay_t<decltype(input)>>(input));
    },
    std::move(color));
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> TransformStack::convert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  return convert_(
    [&](auto&& transform, auto&& input) {
      return transform->convert_color_shape(std::forward<std::decay_t<decltype(input)>>(input));
    },
    std::move(color_shape));
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> TransformStack::convert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  return convert_(
    [&](auto&& transform, auto&& input) {
      return transform->convert_point(std::forward<std::decay_t<decltype(input)>>(input));
    },
    std::move(point));
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> TransformStack::convert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  return convert_(
    [&](auto&& transform, auto&& input) {
      return transform->convert_extents(std::forward<std::decay_t<decltype(input)>>(input));
    },
    std::move(extents));
}

proj::SymbolicPoint TransformStack::invert(proj::SymbolicPoint point) const
{
  return invert_(
    [&](auto&& transform, auto&& input) {
      return transform->invert(std::forward<std::decay_t<decltype(input)>>(input));
    },
    std::move(point));
}

Restrictions TransformStack::invert(Restrictions restrictions) const
{
  return invert_(
    [&](auto&& transform, auto&& input) {
      return transform->invert(std::forward<std::decay_t<decltype(input)>>(input));
    },
    std::move(restrictions));
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> TransformStack::invert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  return invert_(
    [&](auto&& transform, auto&& input) {
      return transform->invert_color(std::forward<std::decay_t<decltype(input)>>(input));
    },
    std::move(color));
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> TransformStack::invert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  return invert_(
    [&](auto&& transform, auto&& input) {
      return transform->invert_color_shape(std::forward<std::decay_t<decltype(input)>>(input));
    },
    std::move(color_shape));
}

SmallVector<std::int32_t, LEGATE_MAX_DIM> TransformStack::invert_dims(
  SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const
{
  return invert_(
    [&](auto&& transform, auto&& input) {
      return transform->invert_dims(std::forward<std::decay_t<decltype(input)>>(input));
    },
    std::move(dims));
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> TransformStack::invert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  return invert_(
    [&](auto&& transform, auto&& input) {
      return transform->invert_point(std::forward<std::decay_t<decltype(input)>>(input));
    },
    std::move(point));
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> TransformStack::invert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  return invert_(
    [&](auto&& transform, auto&& input) {
      return transform->invert_extents(std::forward<std::decay_t<decltype(input)>>(input));
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

SmallVector<std::int32_t, LEGATE_MAX_DIM> TransformStack::find_imaginary_dims() const
{
  SmallVector<std::int32_t, LEGATE_MAX_DIM> dims;

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

SmallVector<std::int64_t, LEGATE_MAX_DIM> Shift::convert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  point[dim_] += offset_;
  return point;
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Shift::invert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
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

SmallVector<std::int32_t, LEGATE_MAX_DIM> Shift::invert_dims(
  SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const
{
  return dims;
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
  restrictions.insert(restrictions.begin() + extra_dim_,
                      forbid_fake_dim ? Restriction::FORBID : Restriction::AVOID);
  return restrictions;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Promote::convert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  color.insert(color.begin() + extra_dim_, 0);
  return color;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Promote::convert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  color_shape.insert(color_shape.begin() + extra_dim_, 1);
  return color_shape;
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Promote::convert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  point.insert(point.begin() + extra_dim_, 0);
  return point;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Promote::convert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  extents.insert(extents.begin() + extra_dim_, dim_size_);
  return extents;
}

proj::SymbolicPoint Promote::invert(proj::SymbolicPoint point) const
{
  point.remove_inplace(extra_dim_);
  return point;
}

Restrictions Promote::invert(Restrictions restrictions) const
{
  restrictions.erase(restrictions.begin() + extra_dim_);
  return restrictions;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Promote::invert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  color.erase(color.begin() + extra_dim_);
  return color;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Promote::invert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  color_shape.erase(color_shape.begin() + extra_dim_);
  return color_shape;
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Promote::invert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  point.erase(point.begin() + extra_dim_);
  return point;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Promote::invert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  extents.erase(extents.begin() + extra_dim_);
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

void Promote::find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>& dims) const
{
  for (auto&& dim : dims) {
    if (dim >= extra_dim_) {
      dim++;
    }
  }
  dims.push_back(extra_dim_);
}

SmallVector<std::int32_t, LEGATE_MAX_DIM> Promote::invert_dims(
  SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const
{
  // Remove the promoted dimension from the ordering(extra_dim_)
  // and renumber dims > extra_dim_

  const auto it = std::find(dims.begin(), dims.end(), extra_dim_);

  // If the dimension added by this promotion is projected in the transformation stack,
  // the dimension index does not exist in `dims`.
  if (it != dims.end()) {
    dims.erase(it);
  }

  std::transform(
    dims.begin(), dims.end(), dims.begin(), [&](auto d) { return d > extra_dim_ ? d - 1 : d; });

  return dims;
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
  restrictions.erase(restrictions.begin() + dim_);
  return restrictions;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Project::convert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  color.erase(color.begin() + dim_);
  return color;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Project::convert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  color_shape.erase(color_shape.begin() + dim_);
  return color_shape;
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Project::convert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  point.erase(point.begin() + dim_);
  return point;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Project::convert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  extents.erase(extents.begin() + dim_);
  return extents;
}

proj::SymbolicPoint Project::invert(proj::SymbolicPoint point) const
{
  point.insert_inplace(dim_, proj::SymbolicExpr{});
  return point;
}

Restrictions Project::invert(Restrictions restrictions) const
{
  restrictions.insert(restrictions.begin() + dim_, Restriction::ALLOW);
  return restrictions;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Project::invert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  color.insert(color.begin() + dim_, 0);
  return color;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Project::invert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  color_shape.insert(color_shape.begin() + dim_, 1);
  return color_shape;
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Project::invert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  point.insert(point.begin() + dim_, coord_);
  return point;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Project::invert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  extents.insert(extents.begin() + dim_, 1);
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

void Project::find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>& dims) const
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

SmallVector<std::int32_t, LEGATE_MAX_DIM> Project::invert_dims(
  SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const
{
  // Renumber dims > projected dim_
  std::transform(
    dims.begin(), dims.end(), dims.begin(), [&](auto&& d) { return d >= dim_ ? d + 1 : d; });

  // Ordering for the projected dimension is left unspecified because the input dimension ordering
  // carries no information to determine the right ordering for it.
  //
  // The partial ordering also allows the mapper to reuse instances of the original store to map the
  // store with projected dimensions. Here's an example to understand why this is the case: Suppose
  // we have a 2D store constructed from a 3D store by projecting the first dimension and want to
  // create an instance with C ordering. The `invert_dims()` call would get an ordering (1, 0) and
  // generate (2, 1), ultimately mapped into (LEGION_DIM_Z, LEGION_DIM_Y). This partial ordering is
  // consistent with the original store's C ordering (LEGION_DIM_Z, LEGION_DIM_Y, LEGION_DIM_X), and
  // thus the projected store can reuse any C instance created for the original store. Let's now say
  // that we added the projected dimension as the first one like the code was previously doing.
  // Then, the dimension ordering we get would be (LEGION_DIM_X, LEGION_DIM_Z, LEGION_DIM_Y),
  // which it contradicts the C ordering of the original store.
  //
  // Technically, we can filter out any unit-extent dimensions from the dimension orderings before
  // we do an entailment check, but that would make the entailment check fairly complicated for no
  // meaningful benefits.

  return dims;
}

// ==========================================================================================

Domain DimBroadcast::transform(const Domain& input) const
{
  Domain output;
  output.dim = input.dim;

  for (std::int32_t dim = 0; dim < output.dim; ++dim) {
    if (dim == dim_) {
      output.rect_data[dim]              = 0;
      output.rect_data[dim + output.dim] = static_cast<coord_t>(dim_size_) - 1;
    } else {
      output.rect_data[dim]              = input.rect_data[dim];
      output.rect_data[dim + output.dim] = input.rect_data[dim + input.dim];
    }
  }
  return output;
}

Legion::DomainAffineTransform DimBroadcast::inverse_transform(std::int32_t in_dim) const
{
  LEGATE_CHECK(dim_ < in_dim);
  const auto out_dim = in_dim;
  Legion::DomainAffineTransform result;

  result.transform.m = out_dim;
  result.transform.n = in_dim;
  for (std::int32_t i = 0; i < result.transform.m; ++i) {
    for (std::int32_t j = 0; j < result.transform.n; ++j) {
      result.transform.matrix[(i * in_dim) + j] = static_cast<std::int32_t>(i == j);
    }
  }
  result.transform.matrix[(dim_ * in_dim) + dim_] = 0;

  result.offset.dim = out_dim;
  for (std::int32_t i = 0; i < result.transform.m; ++i) {
    result.offset[i] = 0;
  }

  return result;
}

Restrictions DimBroadcast::convert(Restrictions restrictions, bool forbid_fake_dim) const
{
  restrictions.at(dim_) = forbid_fake_dim ? Restriction::FORBID : Restriction::AVOID;
  return restrictions;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> DimBroadcast::convert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  LEGATE_ASSERT(extents[dim_] == 1);
  extents.at(dim_) = dim_size_;
  return extents;
}

proj::SymbolicPoint DimBroadcast::invert(proj::SymbolicPoint point) const
{
  point.at(dim_) = constant(0);
  return point;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> DimBroadcast::invert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  color.at(dim_) = 0;
  return color;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> DimBroadcast::invert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  color_shape.at(dim_) = 1;
  return color_shape;
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> DimBroadcast::invert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  point.at(dim_) = 0;
  return point;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> DimBroadcast::invert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  extents.at(dim_) = 1;
  return extents;
}

void DimBroadcast::pack(BufferBuilder& buffer) const
{
  buffer.pack<CoreTransform>(CoreTransform::BROADCAST);
  buffer.pack<std::int32_t>(dim_);
  buffer.pack<std::uint64_t>(dim_size_);
}

void DimBroadcast::print(std::ostream& out) const
{
  out << "Broadcast(";
  out << "dim: " << dim_ << ", ";
  out << "dim_size: " << dim_size_ << ")";
}

void DimBroadcast::find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>& dims) const
{
  dims.push_back(dim_);
}

// ==========================================================================================

Transpose::Transpose(SmallVector<std::int32_t, LEGATE_MAX_DIM>&& axes) : axes_{std::move(axes)}
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
  return array_map(restrictions, axes_);
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Transpose::convert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  // No in-place available
  return array_map(color, axes_);
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Transpose::convert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  // No in-place available
  return array_map(color_shape, axes_);
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Transpose::convert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  // No in-place available
  return array_map(point, axes_);
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Transpose::convert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  // No in-place available
  return array_map(extents, axes_);
}

proj::SymbolicPoint Transpose::invert(proj::SymbolicPoint point) const
{
  // No in-place available
  return point.map(inverse_);
}

Restrictions Transpose::invert(Restrictions restrictions) const
{
  // No in-place available
  return array_map(restrictions, inverse_);
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Transpose::invert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  // No in-place available
  return array_map(color, inverse_);
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Transpose::invert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  // No in-place available
  return array_map(color_shape, inverse_);
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Transpose::invert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  // No in-place available
  return array_map(point, inverse_);
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Transpose::invert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  // No in-place available
  return array_map(extents, inverse_);
}

namespace {

template <typename T>
void print_vector(std::ostream& out, const SmallVector<T, LEGATE_MAX_DIM>& vec)
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

}  // namespace

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

void Transpose::find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>& dims) const
{
  // i should be added to X.transpose(axes).promoted iff axes[i] is in X.promoted
  // e.g. X.promoted = [0] => X.transpose((1,2,0)).promoted = [2]
  for (auto&& promoted : dims) {
    auto finder = std::find(axes_.begin(), axes_.end(), promoted);

    LEGATE_CHECK(finder != axes_.end());
    promoted = static_cast<std::int32_t>(finder - axes_.begin());
  }
}

SmallVector<std::int32_t, LEGATE_MAX_DIM> Transpose::invert_dims(
  SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const
{
  decltype(dims) ret;

  ret.reserve(dims.size());
  std::transform(dims.begin(), dims.end(), std::back_inserter(ret), [&](std::int32_t dim) {
    // Unlike points or shapes, whose indices correspond to dimension indices, `dims` has dimension
    // indices as values, so the inversion is simply looking up the `axes_` using those values.
    return axes_[dim];
  });
  return ret;
}

// ==========================================================================================

Delinearize::Delinearize(std::int32_t dim, SmallVector<std::uint64_t, LEGATE_MAX_DIM>&& sizes)
  : dim_{dim}, sizes_{std::move(sizes)}, strides_{tags::size_tag, sizes_.size(), 1}, volume_{1}
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
    result.push_back(restrictions[dim]);
  }
  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    result.push_back(Restriction::FORBID);
  }
  for (std::uint32_t dim = dim_ + 1; dim < restrictions.size(); ++dim) {
    result.push_back(restrictions[dim]);
  }
  return result;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Delinearize::convert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> /*color*/) const
{
  throw TracedException<NonInvertibleTransformation>{};
  return {};
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Delinearize::convert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> /*color_shape*/) const
{
  throw TracedException<NonInvertibleTransformation>{};
  return {};
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Delinearize::convert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> /*extents*/) const
{
  throw TracedException<NonInvertibleTransformation>{};
  return {};
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Delinearize::convert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> /*point*/) const
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
    result.push_back(restrictions[dim]);
  }

  for (auto dim = dim_ + sizes_.size(); dim < restrictions.size(); ++dim) {
    result.push_back(restrictions[dim]);
  }
  return result;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Delinearize::invert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  auto sum = std::uint64_t{0};
  for (std::uint64_t idx = 1; idx < sizes_.size(); ++idx) {
    sum += color[dim_ + idx];
  }

  if (sum != 0) {
    throw TracedException<NonInvertibleTransformation>{};
  }

  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    color.erase(color.begin() + dim_ + 1);
  }

  return color;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Delinearize::invert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  auto volume = std::uint64_t{1};
  for (std::uint64_t idx = 1; idx < sizes_.size(); ++idx) {
    volume *= color_shape[static_cast<std::uint64_t>(dim_) + idx];
  }

  if (volume != 1) {
    throw TracedException<NonInvertibleTransformation>{};
  }

  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    color_shape.erase(color_shape.begin() + dim_ + 1);
  }
  return color_shape;
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Delinearize::invert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> point) const
{
  auto sum = std::int64_t{0};
  for (std::uint64_t idx = 1; idx < sizes_.size(); ++idx) {
    sum += point[static_cast<std::uint64_t>(dim_) + idx];
  }

  if (sum != 0) {
    throw TracedException<NonInvertibleTransformation>{};
  }

  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    point.erase(point.begin() + dim_ + 1);
  }
  point[static_cast<std::uint64_t>(dim_)] *= static_cast<std::int64_t>(strides_[0]);

  return point;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Delinearize::invert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  for (std::uint64_t idx = 1; idx < sizes_.size(); ++idx) {
    if (extents[static_cast<std::uint64_t>(dim_) + idx] != sizes_[idx]) {
      throw TracedException<NonInvertibleTransformation>{};
    }
  }

  for (std::uint32_t idx = 1; idx < sizes_.size(); ++idx) {
    extents.erase(extents.begin() + dim_ + 1);
  }
  extents[static_cast<std::uint64_t>(dim_)] *= strides_[0];

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

SmallVector<std::int32_t, LEGATE_MAX_DIM> Delinearize::invert_dims(
  SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const
{
  // Collapse the delinearized dimensions back to the original dimension
  const auto num_extra_dims = static_cast<std::int32_t>(sizes_.size()) - 1;
  auto new_end              = std::remove_if(dims.begin(), dims.end(), [&](const auto& d) {
    return (d > dim_) && (d <= (dim_ + num_extra_dims));
  });

  LEGATE_ASSERT(new_end != dims.end());
  dims.erase(new_end, dims.end());

  std::transform(dims.begin(), dims.end(), dims.begin(), [&](const auto& d) {
    return (d > (dim_ + num_extra_dims)) ? (d - num_extra_dims) : d;
  });

  return dims;
}

std::ostream& operator<<(std::ostream& out, const Transform& transform)
{
  transform.print(out);
  return out;
}

}  // namespace legate::detail
