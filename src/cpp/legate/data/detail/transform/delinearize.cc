/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/delinearize.h>

#include <legate/data/detail/transform/non_invertible_transformation.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/typedefs.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <utility>

namespace legate::detail {

Delinearize::Delinearize(std::int32_t dim, SmallVector<std::uint64_t, LEGATE_MAX_DIM>&& sizes)
  : dim_{dim},
    sizes_{std::move(sizes)},
    strides_{tags::size_tag, sizes_.size(), 1},
    volume_{array_volume(sizes_)}
{
  // Need this double cast since sizes_.size() might be < 2, and since the condition is >= 0,
  // we cannot just use std::size_t here
  for (auto size_dim = static_cast<std::int32_t>(sizes_.size()) - 2; size_dim >= 0; --size_dim) {
    const auto usize_dim = static_cast<std::size_t>(size_dim);

    strides_[usize_dim] = strides_[usize_dim + 1] * sizes_[usize_dim + 1];
  }
}

Domain Delinearize::transform(const Domain& domain) const
{
  const auto domain_dim = domain.get_dim();
  Domain output;

  output.dim = static_cast<std::int32_t>(domain_dim - 1 + strides_.size());
  for (std::int32_t in_dim = 0, out_dim = 0; in_dim < domain_dim; ++in_dim) {
    if (in_dim == dim_) {
      auto lo = domain.rect_data[in_dim];
      auto hi = domain.rect_data[domain_dim + in_dim];

      for (auto stride : strides_) {
        const auto c_stride = static_cast<coord_t>(stride);

        output.rect_data[out_dim]              = lo / c_stride;
        output.rect_data[output.dim + out_dim] = hi / c_stride;
        lo                                     = lo % c_stride;
        hi                                     = hi % c_stride;
        ++out_dim;
      }
    } else {
      output.rect_data[out_dim]              = domain.rect_data[in_dim];
      output.rect_data[output.dim + out_dim] = domain.rect_data[domain_dim + in_dim];
      ++out_dim;
    }
  }
  return output;
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
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Delinearize::convert_color_shape(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> /*color_shape*/) const
{
  throw TracedException<NonInvertibleTransformation>{};
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> Delinearize::convert_extents(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> /*extents*/) const
{
  throw TracedException<NonInvertibleTransformation>{};
}

SmallVector<std::int64_t, LEGATE_MAX_DIM> Delinearize::convert_point(
  SmallVector<std::int64_t, LEGATE_MAX_DIM> /*point*/) const
{
  throw TracedException<NonInvertibleTransformation>{};
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
  buffer.pack<std::uint32_t>(static_cast<std::uint32_t>(sizes_.size()));
  for (auto extent : sizes_) {
    buffer.pack<std::uint64_t>(extent);
  }
}

void Delinearize::print(std::ostream& out) const
{
  out << fmt::format("Delinearize(dim: {}, sizes: {})", dim_, fmt::join(sizes_, ", "));
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
  const auto new_end        = std::remove_if(dims.begin(), dims.end(), [&](std::int32_t d) {
    return (d > dim_) && (d <= (dim_ + num_extra_dims));
  });

  LEGATE_ASSERT(new_end != dims.end());
  dims.erase(new_end, dims.end());

  std::transform(dims.begin(), dims.end(), dims.begin(), [&](std::int32_t d) {
    return (d > (dim_ + num_extra_dims)) ? (d - num_extra_dims) : d;
  });

  return dims;
}

}  // namespace legate::detail
