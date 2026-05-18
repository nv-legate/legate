/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/restriction.h>

#include <legate/data/detail/shape.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/partitioning/detail/partition/opaque.h>
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

Restrictions::Restrictions(SmallVector<Restriction> dimension_restrictions,
                           SmallVector<std::uint64_t, LEGATE_MAX_DIM> minimum_extents,
                           bool require_invertible)
  : req_invertible_{require_invertible},
    dim_restrictions_{std::move(dimension_restrictions)},
    min_extents_{std::move(minimum_extents)}
{
  cache_needs_minimum_extent_check_();
}

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

void Restrictions::apply_minimum_extents(Span<const std::uint64_t> new_minimum_extents)
{
  LEGATE_ASSERT(new_minimum_extents.size() == min_extents_.size());

  for (auto&& [dim, new_min_ext] : enumerate(new_minimum_extents)) {
    min_extents_[dim] = std::max<>(min_extents_[dim], new_min_ext);
  }

  cache_needs_minimum_extent_check_();
}

namespace {

[[nodiscard]] constexpr bool smallest_extent_smaller_than_min_extent(
  const std::uint64_t& extent, const std::uint64_t& num_colors, const std::uint64_t& min_extent)
{
  const auto tile_extent     = (extent + num_colors - 1) / num_colors;
  const auto smallest_extent = extent - ((num_colors - 1) * tile_extent);

  return smallest_extent < min_extent;
}

}  // namespace

bool Restrictions::are_satisfied_by(const Partition& partition,
                                    const InternalSharedPtr<Shape>& shape) const
{
  if (require_invertible_() && !partition.is_invertible()) {
    return false;
  }

  if (!partition.has_color_shape()) {
    // Currently NoPartition always has no color shape and Opaque can sometime have no color shape
    LEGATE_ASSERT(dynamic_cast<const NoPartition*>(&partition) ||
                  dynamic_cast<const Opaque*>(&partition));
    return true;
  }

  LEGATE_ASSERT(!needs_minimum_extent_check_ || shape);

  // The `needs_minimum_extent_check_` predicate guards the blocking extent lookup on the deferred
  // shape
  if (needs_minimum_extent_check_ && array_any_of(smallest_extent_smaller_than_min_extent,
                                                  shape->extents(),
                                                  partition.color_shape(),
                                                  minimum_extents_())) {
    return false;
  }

  constexpr auto satisfies_restriction = [](Restriction r, std::uint64_t ext) {
    return r != Restriction::FORBID || ext == 1;
  };

  return array_all_of(satisfies_restriction, dimension_restrictions_(), partition.color_shape());
}

bool Restrictions::minimum_extents_satisfied_by(Span<const std::uint64_t> shape,
                                                Span<const std::uint64_t> color_shape) const
{
  if (!needs_minimum_extent_check_) {
    return true;
  }

  return !array_any_of(
    smallest_extent_smaller_than_min_extent, shape, color_shape, minimum_extents_());
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

  if (other.needs_minimum_extent_check_) {
    apply_minimum_extents(other.minimum_extents_());
  }
}

[[nodiscard]] std::tuple<SmallVector<std::size_t, LEGATE_MAX_DIM>,
                         SmallVector<std::uint32_t, LEGATE_MAX_DIM>,
                         std::uint64_t>
Restrictions::prune_dimensions(Span<const std::uint64_t> shape) const
{
  // Prune out any dimensions that are 1
  SmallVector<std::size_t, LEGATE_MAX_DIM> temp_shape{};
  SmallVector<std::uint32_t, LEGATE_MAX_DIM> temp_dims{};
  std::uint64_t volume = 1;

  temp_dims.reserve(shape.size());
  temp_shape.reserve(shape.size());
  for (auto&& [dim, rest] : enumerate(zip_equal(dimension_restrictions_(), shape))) {
    auto&& [restr, extent] = rest;

    if (1 == extent || restr == Restriction::FORBID) {
      continue;
    }
    temp_shape.push_back(extent);
    temp_dims.push_back(dim);
    volume *= extent;
  }
  return {std::move(temp_shape), std::move(temp_dims), volume};
}

Legion::Domain Restrictions::prune_dimensions(const Legion::Domain& domain) const
{
  auto new_lo   = Legion::DomainPoint{};
  auto new_hi   = Legion::DomainPoint{};
  auto new_ndim = std::uint32_t{0};

  for (auto&& [dim, restriction] : enumerate(dimension_restrictions_())) {
    if (restriction == Restriction::FORBID) {
      continue;
    }
    new_lo[new_ndim] = domain.rect_data[dim];
    new_hi[new_ndim] = domain.rect_data[dim + domain.dim];
    ++new_ndim;
  }

  new_lo.dim = static_cast<std::int32_t>(new_ndim);
  new_hi.dim = static_cast<std::int32_t>(new_ndim);

  return {new_lo, new_hi};
}

Legion::Domain Restrictions::embed(const Legion::Domain& domain) const
{
  auto new_lo = Legion::DomainPoint{};
  auto new_hi = Legion::DomainPoint{};
  auto idx    = std::uint32_t{0};

  new_lo.dim = static_cast<std::int32_t>(dimension_restrictions_().size());
  new_hi.dim = static_cast<std::int32_t>(dimension_restrictions_().size());
  for (auto&& [dim, restriction] : enumerate(dimension_restrictions_())) {
    if (restriction == Restriction::FORBID) {
      new_lo[dim] = 0;
      new_hi[dim] = 0;
      continue;
    }
    new_lo[dim] = domain.rect_data[idx];
    new_hi[dim] = domain.rect_data[idx + domain.dim];
    ++idx;
  }

  return {new_lo, new_hi};
}

std::size_t Restrictions::count_restricted() const
{
  return std::count_if(
    dimension_restrictions_().begin(),
    dimension_restrictions_().end(),
    [](auto&& restriction) noexcept { return restriction == Restriction::FORBID; });
}

SymbolicPoint Restrictions::to_projection() const
{
  auto dim   = std::uint32_t{0};
  auto point = SymbolicPoint{};

  point.reserve(dimension_restrictions_().size());
  for (auto&& restriction : dimension_restrictions_()) {
    point.append_inplace(restriction == Restriction::FORBID ? constant(0) : dimension(dim++));
  }

  return point;
}

void Restrictions::cache_needs_minimum_extent_check_()
{
  needs_minimum_extent_check_ = std::any_of(minimum_extents_().begin(),
                                            minimum_extents_().end(),
                                            [](auto&& min_ext) noexcept { return min_ext != 0; });
}

}  // namespace legate::detail
