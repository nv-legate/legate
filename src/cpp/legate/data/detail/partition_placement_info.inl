/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/partition_placement_info.h>

namespace legate::detail {

inline PartitionPlacementInfo::PartitionPlacementInfo(std::vector<PartitionPlacement> placements)
  : placements_(std::move(placements))
{
}

inline Span<const PartitionPlacement> PartitionPlacementInfo::placements() const
{
  return placements_;
}

}  // namespace legate::detail
