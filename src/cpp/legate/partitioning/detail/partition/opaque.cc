/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partition/opaque.h>

#include <legate/data/detail/transform/non_invertible_transformation.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace legate::detail {

Opaque::Opaque(Legion::IndexSpace ispace,
               Legion::IndexPartition ipartition,
               const Domain& color_domain)
  : ispace_{std::move(ispace)},
    ipartition_{std::move(ipartition)},
    color_domain_{color_domain},
    color_shape_{detail::from_domain(color_domain)}
{
}

bool Opaque::operator==(const Opaque& other) const
{
  // Equality check on index partitions is sufficient
  return ipartition_ == other.ipartition_;
}

bool Opaque::operator<(const Opaque& other) const { return ipartition_ < other.ipartition_; }

bool Opaque::is_disjoint_for(const Domain& launch_domain) const
{
  // TODO(wonchanl): The check really should be that every two points from the launch domain are
  // mapped to two different colors
  return !launch_domain.is_valid() || launch_domain.get_volume() <= color_domain_.get_volume();
}

InternalSharedPtr<Partition> Opaque::scale(Span<const std::uint64_t> /*factors*/) const
{
  throw TracedException<std::runtime_error>{"Not implemented"};
  return {};
}

InternalSharedPtr<Partition> Opaque::bloat(Span<const std::uint64_t> /*low_offsts*/,
                                           Span<const std::uint64_t> /*high_offsets*/) const
{
  throw TracedException<std::runtime_error>{"Not implemented"};
  return {};
}

Legion::LogicalPartition Opaque::construct(Legion::LogicalRegion region, bool) const
{
  auto&& runtime     = detail::Runtime::get_runtime();
  const auto& target = region.get_index_space();
  if (target == ispace_) {
    return runtime.create_logical_partition(region, ipartition_);
  }

  return runtime.create_logical_partition(
    region, runtime.create_intersection_partition(target, ipartition_));
}

std::string Opaque::to_string() const
{
  return fmt::format(
    "Opaque(IndexPartition({}, {}))", ipartition_.get_id(), ipartition_.get_tree_id());
}

InternalSharedPtr<Partition> Opaque::convert(
  const InternalSharedPtr<Partition>& self,
  const InternalSharedPtr<TransformStack>& transform) const
{
  if (transform->identity()) {
    return self;
  }
  throw TracedException<std::runtime_error>{
    "We can not convert an Opaque Partition without identity transformation"};
}

InternalSharedPtr<Partition> Opaque::invert(
  const InternalSharedPtr<Partition>& self,
  const InternalSharedPtr<TransformStack>& transform) const
{
  if (transform->identity()) {
    return self;
  }
  throw TracedException<NonInvertibleTransformation>{};
}

// ==========================================================================================

InternalSharedPtr<Opaque> create_opaque(Legion::IndexSpace ispace,
                                        Legion::IndexPartition ipartition,
                                        const Domain& color_domain)
{
  return make_internal_shared<Opaque>(std::move(ispace), std::move(ipartition), color_domain);
}

}  // namespace legate::detail
