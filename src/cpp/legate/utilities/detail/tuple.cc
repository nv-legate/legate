/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/tuple.h>

#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstdint>
#include <stdexcept>

namespace legate::detail {

Domain to_domain(Span<const std::uint64_t> shape)
{
  if (shape.empty()) {
    return {0, 0};
  }

  const auto ndim = static_cast<std::uint32_t>(shape.size());
  Domain domain;

  domain.dim = static_cast<int>(ndim);
  for (std::uint32_t idx = 0; idx < ndim; ++idx) {
    domain.rect_data[idx]        = 0;
    domain.rect_data[idx + ndim] = static_cast<coord_t>(shape[idx]) - 1;
  }
  return domain;
}

DomainPoint to_domain_point(Span<const std::uint64_t> shape)
{
  const auto ndim = static_cast<std::uint32_t>(shape.size());
  DomainPoint point;

  point.dim = static_cast<int>(ndim);
  for (std::uint32_t idx = 0; idx < ndim; ++idx) {
    point[idx] = static_cast<coord_t>(shape[idx]);
  }
  return point;
}

SmallVector<std::uint64_t, LEGATE_MAX_DIM> from_domain(const Domain& domain)
{
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> result;
  auto&& lo = domain.lo();
  auto&& hi = domain.hi();

  result.reserve(domain.dim);
  for (std::int32_t idx = 0; idx < domain.dim; ++idx) {
    result.emplace_back(hi[idx] - lo[idx] + 1);
  }
  return result;
}

void throw_invalid_tuple_sizes(std::size_t lhs_size, std::size_t rhs_size)
{
  throw TracedException<std::invalid_argument>{
    fmt::format("Operands should have the same size: {} != {}", lhs_size, rhs_size)};
}

}  // namespace legate::detail
