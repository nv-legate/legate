/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/geometry.h>

#include <legate/utilities/assert.h>

#include <cstdint>

namespace legate::detail {

std::int64_t inner_product(const DomainPoint& rhs1, const DomainPoint& rhs2) noexcept
{
  LEGATE_ASSERT(rhs1.dim == rhs2.dim);

  std::int64_t result = 0;

  for (std::int32_t dim = 0; dim < rhs1.dim; ++dim) {
    result += rhs1[dim] * rhs2[dim];
  }

  return result;
}

std::int64_t calculate_volume(const DomainPoint& shape) noexcept
{
  std::int64_t result = 1;

  for (std::int32_t dim = 0; dim < shape.dim; ++dim) {
    result *= shape[dim];
  }

  return result;
}

DomainPoint calculate_strides(const Domain& domain) noexcept
{
  DomainPoint strides;

  strides.dim = domain.dim;
  // TODO(wonchanl): C order is assumed but the layout can be arbitrary
  strides[strides.dim - 1] = 1;
  for (std::int32_t dim = strides.dim - 2; dim >= 0; --dim) {
    strides[dim] =
      strides[dim + 1] * (domain.rect_data[dim + 1 + domain.dim] - domain.rect_data[dim + 1] + 1);
  }

  return strides;
}

DomainPoint calculate_strides(const DomainPoint& shape) noexcept
{
  DomainPoint strides;

  strides.dim = shape.dim;
  // TODO(wonchanl): C order is assumed but the layout can be arbitrary
  strides[strides.dim - 1] = 1;
  for (std::int32_t dim = strides.dim - 2; dim >= 0; --dim) {
    strides[dim] = strides[dim + 1] * shape[dim + 1];
  }

  return strides;
}

DomainPoint to_shape(const Domain& domain) noexcept
{
  DomainPoint shape;

  shape.dim = domain.dim;
  for (std::int32_t dim = 0; dim < shape.dim; ++dim) {
    shape[dim] = domain.rect_data[dim + domain.dim] - domain.rect_data[dim] + 1;
  }

  return shape;
}

}  // namespace legate::detail
