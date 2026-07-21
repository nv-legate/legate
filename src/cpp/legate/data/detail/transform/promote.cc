/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/promote.h>

#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/typedefs.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <utility>

namespace legate::detail {

Domain Promote::transform(const Domain& input) const
{
  const auto out_ndim = input.dim() + 1;
  auto lo             = DomainPoint::zeroes(out_ndim);
  auto hi             = DomainPoint::zeroes(out_ndim);
  auto&& in_lo        = input.lo();
  auto&& in_hi        = input.hi();

  for (std::uint32_t out_dim = 0, in_dim = 0; out_dim < out_ndim; ++out_dim) {
    if (static_cast<std::int32_t>(out_dim) == extra_dim_) {
      lo[out_dim] = 0;
      hi[out_dim] = dim_size_ - 1;
    } else {
      lo[out_dim] = in_lo[in_dim];
      hi[out_dim] = in_hi[in_dim];
      ++in_dim;
    }
  }
  return {std::move(lo), std::move(hi)};
}

Restrictions Promote::convert(Restrictions restrictions, bool forbid_fake_dim) const
{
  return std::move(restrictions)
    .map(
      [&](auto&& dim_res) {
        dim_res.insert(dim_res.begin() + extra_dim_,
                       forbid_fake_dim ? Restriction::FORBID : Restriction::AVOID);
        return dim_res;
      },
      [&](auto&& min_exts) {
        min_exts.insert(min_exts.begin() + extra_dim_, 0);
        return min_exts;
      });
}

DomainPoint Promote::convert_color(DomainPoint color) const
{
  return color.inserted(extra_dim_, /*value=*/0);
}

DomainPoint Promote::convert_color_shape(DomainPoint color_shape) const
{
  return color_shape.inserted(extra_dim_, /*value=*/1);
}

DomainPoint Promote::convert_point(DomainPoint point) const
{
  return point.inserted(extra_dim_, /*value=*/0);
}

DomainPoint Promote::convert_extents(DomainPoint extents) const
{
  return extents.inserted(extra_dim_, dim_size_);
}

DomainPoint Promote::convert_strides(DomainPoint strides) const
{
  // The promoted dimension is not backed by the storage, so moving along it does not move in
  // memory: its stride is 0.
  return strides.inserted(extra_dim_, /*value=*/0);
}

SymbolicPoint Promote::invert(SymbolicPoint point) const
{
  point.remove_inplace(extra_dim_);
  return point;
}

Restrictions Promote::invert(Restrictions restrictions) const
{
  return std::move(restrictions)
    .map(
      [&](auto&& dim_res) {
        dim_res.erase(dim_res.begin() + extra_dim_);
        return dim_res;
      },
      [&](auto&& min_exts) {
        min_exts.erase(min_exts.begin() + extra_dim_);
        return min_exts;
      });
}

DomainPoint Promote::invert_color(DomainPoint color) const { return color.removed(extra_dim_); }

DomainPoint Promote::invert_color_shape(DomainPoint color_shape) const
{
  return color_shape.removed(extra_dim_);
}

DomainPoint Promote::invert_point(DomainPoint point) const { return point.removed(extra_dim_); }

DomainPoint Promote::invert_extents(DomainPoint extents) const
{
  return extents.removed(extra_dim_);
}

void Promote::pack(BufferBuilder& buffer) const
{
  buffer.pack<CoreTransform>(CoreTransform::PROMOTE);
  buffer.pack<std::int32_t>(extra_dim_);
  buffer.pack<std::int64_t>(dim_size_);
}

void Promote::print(std::ostream& out) const
{
  out << fmt::format("Promote(extra_dim: {}, dim_size: {})", extra_dim_, dim_size_);
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

}  // namespace legate::detail
