/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/tuple.h>

#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace legate::detail {

Domain to_domain(Span<const std::uint64_t> shape)
{
  if (shape.size() == 0) {
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

Domain to_domain(const tuple<std::uint64_t>& shape) { return to_domain(shape.data()); }

DomainPoint to_domain_point(const tuple<std::uint64_t>& shape)
{
  const auto ndim = static_cast<std::uint32_t>(shape.size());
  DomainPoint point;

  point.dim = static_cast<int>(ndim);
  for (std::uint32_t idx = 0; idx < ndim; ++idx) {
    point[idx] = static_cast<coord_t>(shape[idx]);
  }
  return point;
}

tuple<std::uint64_t> from_domain(const Domain& domain)
{
  std::vector<std::uint64_t> result;
  auto&& lo = domain.lo();
  auto&& hi = domain.hi();

  result.reserve(domain.dim);
  for (std::int32_t idx = 0; idx < domain.dim; ++idx) {
    result.emplace_back(hi[idx] - lo[idx] + 1);
  }
  return tuple<std::uint64_t>{std::move(result)};
}

void assert_valid_mapping(std::size_t tuple_size, const std::vector<std::int32_t>& mapping)
{
  if (mapping.size() != tuple_size) {
    throw TracedException<std::out_of_range>{
      fmt::format("mapping size {} != tuple size {}", mapping.size(), tuple_size)};
  }

  // Early out here because we use front() and back() below
  if (mapping.empty()) {
    return;
  }

  auto sorted_mapping = mapping;

  std::sort(sorted_mapping.begin(), sorted_mapping.end());
  // Check that elements are in range. The copy is sorted, so it suffices to check the
  // bounds. If either is out of range, then at least one element of the mapping is out of
  // range.
  if (sorted_mapping.front() < 0) {
    throw TracedException<std::out_of_range>{
      fmt::format("mapping {} contains negative elements", mapping)};
  }
  if (static_cast<std::size_t>(sorted_mapping.back()) >= tuple_size) {
    throw TracedException<std::out_of_range>{
      fmt::format("mapping {} contains elements outside of tuple size {}", mapping, tuple_size)};
  }

  // Check that elements are unique
  if (const auto it = std::adjacent_find(sorted_mapping.begin(), sorted_mapping.end());
      it != sorted_mapping.end()) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Invalid mapping: contains duplicate element(s) {} ({})", *it, mapping)};
  }
}

void throw_invalid_tuple_sizes(std::size_t lhs_size, std::size_t rhs_size)
{
  throw TracedException<std::invalid_argument>{
    fmt::format("Operands should have the same size: {} != {}", lhs_size, rhs_size)};
}

void assert_in_range(std::size_t tuple_size, std::int32_t pos)
{
  if ((pos < 0) || (static_cast<std::size_t>(pos) >= tuple_size)) {
    throw TracedException<std::out_of_range>{
      fmt::format("Index {} out of range [0, {})", pos, tuple_size)};
  }
}

}  // namespace legate::detail
