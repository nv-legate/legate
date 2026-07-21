/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

Rect to_domain(Span<const std::uint64_t> shape)
{
  if (shape.empty()) {
    return {0, 0};
  }

  const auto ndim = static_cast<std::int32_t>(shape.size());
  auto lo         = Point::zeroes(ndim);
  auto hi         = Point::zeroes(ndim);

  for (std::int32_t idx = 0; idx < ndim; ++idx) {
    hi[idx] = static_cast<std::int64_t>(shape[idx]) - 1;
  }
  return {std::move(lo), std::move(hi)};
}

Rect to_domain(const Point& shape)
{
  if (shape.empty()) {
    return {0, 0};
  }
  return {Point::zeroes(static_cast<std::int32_t>(shape.size())), shape - 1};
}

Rect to_maybe_empty_domain(Span<const std::uint64_t> shape)
{
  // When empty domains are serialized, they come out as (0-ed out), 0D Domains
  // (e.g. when deserialized). However, to_domain() will produce a {0, 0} domain,
  // i.e. still 0-ed out, but this time it is 1D. So to match these cases,
  // we return a empty 0D domain here.
  if (shape.empty()) {
    return {};
  }
  return to_domain(shape);
}

Point to_domain_point(Span<const std::uint64_t> shape)
{
  const auto ndim = static_cast<std::int32_t>(shape.size());
  auto point      = Point::zeroes(ndim);

  for (std::int32_t idx = 0; idx < ndim; ++idx) {
    point[idx] = static_cast<std::int64_t>(shape[idx]);
  }
  return point;
}

Point to_domain_point(Span<const std::int64_t> shape) { return Point{shape}; }

SmallVector<std::uint64_t, LEGATE_MAX_DIM> from_domain(const Rect& domain)
{
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> result;
  auto&& lo = domain.lo();
  auto&& hi = domain.hi();

  result.reserve(static_cast<std::size_t>(domain.dim()));
  for (std::uint32_t idx = 0; idx < domain.dim(); ++idx) {
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
