/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/project.h>

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

Domain Project::transform(const Domain& input) const
{
  LEGATE_ASSERT(input.dim() > 0);
  const auto out_ndim = input.dim() - 1;
  auto lo             = DomainPoint::zeroes(static_cast<std::int32_t>(out_ndim));
  auto hi             = DomainPoint::zeroes(static_cast<std::int32_t>(out_ndim));
  auto&& in_lo        = input.lo();
  auto&& in_hi        = input.hi();

  for (std::uint32_t in_dim = 0, out_dim = 0; in_dim < input.dim(); ++in_dim) {
    if (static_cast<std::int32_t>(in_dim) != dim_) {
      lo[out_dim] = in_lo[in_dim];
      hi[out_dim] = in_hi[in_dim];
      ++out_dim;
    }
  }
  return {std::move(lo), std::move(hi)};
}

Restrictions Project::convert(Restrictions restrictions, bool /*forbid_fake_dim*/) const
{
  return std::move(restrictions)
    .map(
      [&](auto&& dim_res) {
        dim_res.erase(dim_res.begin() + dim_);
        return dim_res;
      },
      [&](auto&& min_exts) {
        min_exts.erase(min_exts.begin() + dim_);
        return min_exts;
      });
}

DomainPoint Project::convert_color(DomainPoint color) const { return color.removed(dim_); }

DomainPoint Project::convert_color_shape(DomainPoint color_shape) const
{
  return color_shape.removed(dim_);
}

DomainPoint Project::convert_point(DomainPoint point) const { return point.removed(dim_); }

DomainPoint Project::convert_extents(DomainPoint extents) const { return extents.removed(dim_); }

DomainPoint Project::convert_strides(DomainPoint strides) const
{
  // The projected dimension is fixed to a single coordinate; its stride is dropped from the
  // store (the fixed coordinate's contribution is folded into the base pointer instead).
  return strides.removed(dim_);
}

SymbolicPoint Project::invert(SymbolicPoint point) const
{
  point.insert_inplace(dim_, SymbolicExpr{});
  return point;
}

Restrictions Project::invert(Restrictions restrictions) const
{
  return std::move(restrictions)
    .map(
      [&](auto&& dim_res) {
        dim_res.insert(dim_res.begin() + dim_, Restriction::ALLOW);
        return dim_res;
      },
      [&](auto&& min_exts) {
        min_exts.insert(min_exts.begin() + dim_, 0);
        return min_exts;
      });
}

DomainPoint Project::invert_color(DomainPoint color) const
{
  return color.inserted(dim_, /*value=*/0);
}

DomainPoint Project::invert_color_shape(DomainPoint color_shape) const
{
  return color_shape.inserted(dim_, /*value=*/1);
}

DomainPoint Project::invert_point(DomainPoint point) const { return point.inserted(dim_, coord_); }

DomainPoint Project::invert_extents(DomainPoint extents) const
{
  return extents.inserted(dim_, /*value=*/1);
}

void Project::pack(BufferBuilder& buffer) const
{
  buffer.pack<CoreTransform>(CoreTransform::PROJECT);
  buffer.pack<std::int32_t>(dim_);
  buffer.pack<std::int64_t>(coord_);
}

void Project::print(std::ostream& out) const
{
  out << fmt::format("Project(dim: {}, coord: {})", dim_, coord_);
}

void Project::find_imaginary_dims(SmallVector<std::int32_t, LEGATE_MAX_DIM>& dims) const
{
  if (const auto it = std::find(dims.begin(), dims.end(), dim_); it != dims.end()) {
    dims.erase(it);
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

}  // namespace legate::detail
