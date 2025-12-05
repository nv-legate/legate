/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/partition_placement.h>

namespace legate::detail {

inline PartitionPlacement::PartitionPlacement(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> partition_color,
  std::uint32_t node_id,
  mapping::StoreTarget memory_type)
  : partition_color_{std::move(partition_color)}, node_id_{node_id}, memory_type_{memory_type}
{
}

inline Span<const std::uint64_t> PartitionPlacement::partition_color() const
{
  return partition_color_;
}

inline std::uint32_t PartitionPlacement::node_id() const { return node_id_; }

inline mapping::StoreTarget PartitionPlacement::memory_type() const { return memory_type_; }

}  // namespace legate::detail
