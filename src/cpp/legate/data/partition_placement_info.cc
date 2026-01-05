/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/partition_placement_info.h>

#include <legate/data/detail/partition_placement.h>
#include <legate/data/detail/partition_placement_info.h>
#include <legate/data/partition_placement.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/shared_ptr.h>

namespace legate {

PartitionPlacementInfo::PartitionPlacementInfo(
  InternalSharedPtr<detail::PartitionPlacementInfo> impl)
  : impl_{std::move(impl)}
{
}

std::vector<PartitionPlacement> PartitionPlacementInfo::placements() const
{
  // Convert detail::PartitionPlacement objects to public PartitionPlacement objects
  std::vector<PartitionPlacement> result;
  result.reserve(impl_->placements().size());

  for (const auto& detail_mapping : impl_->placements()) {
    auto detail_ptr = make_internal_shared<detail::PartitionPlacement>(
      detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{detail_mapping.partition_color()},
      detail_mapping.node_id(),
      detail_mapping.memory_type());
    result.emplace_back(std::move(detail_ptr));
  }

  return result;
}

std::optional<PartitionPlacement> PartitionPlacementInfo::get_placement_for_color(
  Span<const std::uint64_t> color) const
{
  auto detail_mapping = impl_->get_placement_for_color(color);

  if (detail_mapping.has_value()) {
    const auto& detail_obj = detail_mapping->get();
    auto detail_ptr        = make_internal_shared<detail::PartitionPlacement>(
      detail::SmallVector<std::uint64_t, LEGATE_MAX_DIM>{detail_obj.partition_color()},
      detail_obj.node_id(),
      detail_obj.memory_type());

    return PartitionPlacement{SharedPtr<detail::PartitionPlacement>{std::move(detail_ptr)}};
  }
  return std::nullopt;
}

std::string PartitionPlacementInfo::to_string() const { return impl_->to_string(); }

}  // namespace legate
