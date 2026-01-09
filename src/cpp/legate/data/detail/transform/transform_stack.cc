/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/transform_stack.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/small_vector.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

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

}  // namespace legate::detail
