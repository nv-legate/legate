/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/restriction.h>

#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/zip.h>

#include <algorithm>
#include <stdexcept>

namespace legate::detail {

namespace {

Restriction join_restriction(Restriction lhs, Restriction rhs) { return std::max(lhs, rhs); }

}  // namespace

bool Restrictions::operator==(const Restrictions& other) const
{
  return (require_invertible_() == other.require_invertible_()) &&
         (dimension_restrictions_() == other.dimension_restrictions_());
}

void Restrictions::restrict_all_dimensions()
{
  std::fill(
    dimension_restrictions_().begin(), dimension_restrictions_().end(), Restriction::FORBID);
}

void Restrictions::restrict_dimension(std::uint32_t dim)
{
  // TODO(wonchanl): We want to check the axis eagerly and raise an exception
  // if it is out of bounds
  LEGATE_ASSERT(dim < dimension_restrictions_().size());
  dimension_restrictions_().at(dim) = Restriction::FORBID;
}

bool Restrictions::are_satisfied_by(const Partition& partition) const
{
  if (require_invertible_() && !partition.is_invertible()) {
    return false;
  }

  if (!partition.has_color_shape()) {
    // Currently only NoPartition has no color shape
    LEGATE_ASSERT(dynamic_cast<const NoPartition*>(&partition));
    return true;
  }

  constexpr auto satisfies_restriction = [](Restriction r, std::uint64_t ext) {
    return r != Restriction::FORBID || ext == 1;
  };

  return array_all_of(satisfies_restriction, dimension_restrictions_(), partition.color_shape());
}

Restrictions Restrictions::join(const Restrictions& other) const
{
  auto result = *this;

  result.join_inplace(other);
  return result;
}

void Restrictions::join_inplace(const Restrictions& other)
{
  auto& lhs       = dimension_restrictions_();
  const auto& rhs = other.dimension_restrictions_();

  if (lhs.size() != rhs.size()) {
    throw TracedException<std::invalid_argument>{"Restrictions must have the same size"};
  }

  if (rhs.empty()) {
    return;
  }

  req_invertible_ |= other.require_invertible_();

  if (lhs.empty()) {
    lhs = rhs;
    return;
  }
  for (auto&& [lhsv, rhsv] : detail::zip_equal(lhs, rhs)) {
    lhsv = join_restriction(lhsv, rhsv);
  }
}

[[nodiscard]] std::tuple<SmallVector<std::size_t, LEGATE_MAX_DIM>,
                         SmallVector<std::uint32_t, LEGATE_MAX_DIM>,
                         std::int64_t>
Restrictions::prune_dimensions(Span<const std::uint64_t> shape) const
{
  // Prune out any dimensions that are 1
  SmallVector<std::size_t, LEGATE_MAX_DIM> temp_shape{};
  SmallVector<std::uint32_t, LEGATE_MAX_DIM> temp_dims{};
  std::int64_t volume = 1;

  temp_dims.reserve(shape.size());
  temp_shape.reserve(shape.size());
  for (auto&& [dim, rest] : enumerate(zip_equal(dimension_restrictions_(), shape))) {
    auto&& [restr, extent] = rest;

    if (1 == extent || restr == Restriction::FORBID) {
      continue;
    }
    temp_shape.push_back(extent);
    temp_dims.push_back(dim);
    volume *= static_cast<std::int64_t>(extent);
  }
  return {std::move(temp_shape), std::move(temp_dims), volume};
}

}  // namespace legate::detail
