/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partition/no_partition.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace legate::detail {

InternalSharedPtr<Partition> NoPartition::scale(Span<const std::uint64_t> /*factors*/) const
{
  return create_no_partition();
}

InternalSharedPtr<Partition> NoPartition::bloat(Span<const std::uint64_t> /*low_offsts*/,
                                                Span<const std::uint64_t> /*high_offsets*/) const
{
  return create_no_partition();
}

const Point& NoPartition::color_shape() const
{
  throw TracedException<std::invalid_argument>{"NoPartition doesn't support color_shape"};
}

bool NoPartition::is_equi_size(const Rect&) const { return true; }

std::vector<std::uint64_t> NoPartition::compute_device_ids(const Point&) const
{
  // The whole (single) region is replicated on every device.
  std::vector<std::uint64_t> result(Runtime::get_runtime().node_count());

  std::iota(result.begin(), result.end(), std::uint64_t{0});
  return result;
}

Rect NoPartition::launch_domain() const
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

const InternalSharedPtr<Partition>& create_no_partition()
{
  static const auto result =
    static_pointer_cast<Partition, NoPartition>(make_internal_shared<NoPartition>());

  return result;
}

}  // namespace legate::detail
