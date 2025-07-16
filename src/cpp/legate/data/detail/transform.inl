/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/transform.h>

#include <utility>

namespace legate::detail {

inline NonInvertibleTransformation::NonInvertibleTransformation(std::string_view error_message)
  : runtime_error{error_message.data()}  // NOLINT(bugprone-suspicious-stringview-data-usage)
{
}

// ==========================================================================================

inline TransformStack::TransformStack(private_tag,
                                      std::unique_ptr<StoreTransform>&& transform,
                                      InternalSharedPtr<TransformStack> parent)
  : transform_{std::move(transform)},
    parent_{std::move(parent)},
    convertible_{transform_->is_convertible() && parent_->is_convertible()}
{
}

inline TransformStack::TransformStack(std::unique_ptr<StoreTransform>&& transform,
                                      const InternalSharedPtr<TransformStack>& parent)
  : TransformStack{private_tag{}, std::move(transform), parent}
{
}

inline TransformStack::TransformStack(std::unique_ptr<StoreTransform>&& transform,
                                      InternalSharedPtr<TransformStack>&& parent)
  : TransformStack{private_tag{}, std::move(transform), std::move(parent)}
{
}

inline bool TransformStack::is_convertible() const { return convertible_; }

inline bool TransformStack::identity() const { return nullptr == transform_; }

template <typename VISITOR, typename T>
auto TransformStack::convert_(VISITOR visitor, T&& input) const
{
  if (identity()) {
    return input;
  }
  if (parent_->identity()) {
    return visitor(transform_, std::forward<T>(input));
  }
  return visitor(transform_, visitor(parent_, std::forward<T>(input)));
}

template <typename VISITOR, typename T>
auto TransformStack::invert_(VISITOR visitor, T&& input) const
{
  if (identity()) {
    return input;
  }

  auto result = visitor(transform_, std::forward<T>(input));

  if (parent_->identity()) {
    return result;
  }
  return visitor(parent_, std::move(result));
}

// ==========================================================================================

inline Shift::Shift(std::int32_t dim, std::int64_t offset) : dim_{dim}, offset_{offset} {}

// the shift transform makes no change on the store's dimensions
inline proj::SymbolicPoint Shift::invert(proj::SymbolicPoint point) const { return point; }

inline Restrictions Shift::convert(Restrictions restrictions, bool /*forbid_fake_dim*/) const
{
  return restrictions;
}

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM> Shift::convert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  return color;
}

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM> Shift::convert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  return color_shape;
}

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM> Shift::convert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  return extents;
}

inline Restrictions Shift::invert(Restrictions restrictions) const { return restrictions; }

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM> Shift::invert_color(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  return color;
}

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM> Shift::invert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape) const
{
  return color_shape;
}

inline SmallVector<std::uint64_t, LEGATE_MAX_DIM> Shift::invert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents) const
{
  return extents;
}

inline std::int32_t Shift::target_ndim(std::int32_t source_ndim) const { return source_ndim; }

inline void Shift::find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>&) const {}

inline bool Shift::is_convertible() const { return true; }

// ==========================================================================================

inline Promote::Promote(std::int32_t extra_dim, std::int64_t dim_size)
  : extra_dim_{extra_dim}, dim_size_{dim_size}
{
}

inline std::int32_t Promote::target_ndim(std::int32_t source_ndim) const { return source_ndim - 1; }

inline bool Promote::is_convertible() const { return true; }

// ==========================================================================================

inline Project::Project(std::int32_t dim, std::int64_t coord) : dim_{dim}, coord_{coord} {}

inline std::int32_t Project::target_ndim(std::int32_t source_ndim) const { return source_ndim + 1; }

inline bool Project::is_convertible() const { return true; }

// ==========================================================================================

inline std::int32_t Transpose::target_ndim(std::int32_t source_ndim) const { return source_ndim; }

inline bool Transpose::is_convertible() const { return true; }

// ==========================================================================================

inline void Delinearize::find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>&) const {}

inline bool Delinearize::is_convertible() const { return false; }

}  // namespace legate::detail
