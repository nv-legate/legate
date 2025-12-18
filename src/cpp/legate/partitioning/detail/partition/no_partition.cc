/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partition/no_partition.h>

#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace legate::detail {

bool NoPartition::is_disjoint_for(const Domain& launch_domain) const
{
  return !launch_domain.is_valid() || launch_domain.get_volume() == 1;
}

InternalSharedPtr<Partition> NoPartition::scale(Span<const std::uint64_t> /*factors*/) const
{
  return create_no_partition();
}

InternalSharedPtr<Partition> NoPartition::bloat(Span<const std::uint64_t> /*low_offsts*/,
                                                Span<const std::uint64_t> /*high_offsets*/) const
{
  return create_no_partition();
}

Span<const std::uint64_t> NoPartition::color_shape() const
{
  throw TracedException<std::invalid_argument>{"NoPartition doesn't support color_shape"};
}

Legion::Domain NoPartition::launch_domain() const
{
  throw TracedException<std::invalid_argument>{"NoPartition has no launch domain"};
}

std::string NoPartition::to_string() const { return "NoPartition"; }

InternalSharedPtr<Partition> NoPartition::convert(
  const InternalSharedPtr<Partition>& self,
  const InternalSharedPtr<TransformStack>& /*transform*/) const
{
  return self;
}

InternalSharedPtr<Partition> NoPartition::invert(
  const InternalSharedPtr<Partition>& self,
  const InternalSharedPtr<TransformStack>& /*transform*/) const
{
  return self;
}

// ==========================================================================================

InternalSharedPtr<NoPartition> create_no_partition()
{
  static const auto result = make_internal_shared<NoPartition>();

  return result;
}

}  // namespace legate::detail
