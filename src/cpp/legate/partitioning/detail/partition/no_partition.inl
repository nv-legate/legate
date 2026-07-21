/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/partition/no_partition.h>

namespace legate::detail {

inline bool NoPartition::is_convertible() const { return true; }

inline bool NoPartition::is_invertible() const { return true; }

inline bool NoPartition::has_color_shape() const { return false; }

inline Point NoPartition::compute_color(std::uint64_t) const { return {}; }

inline Rect NoPartition::shard(const Rect& domain, const Point&) const { return domain; }

inline bool NoPartition::subsumes(const Rect&, const InternalSharedPtr<Partition>&) const
{
  return true;
}

}  // namespace legate::detail
