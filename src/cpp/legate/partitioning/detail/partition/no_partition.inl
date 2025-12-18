/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/partition/no_partition.h>

namespace legate::detail {

inline NoPartition::Kind NoPartition::kind() const { return Kind::NO_PARTITION; }

inline bool NoPartition::is_complete_for(const detail::Storage& /*storage*/) const { return true; }

inline bool NoPartition::is_convertible() const { return true; }

inline bool NoPartition::is_invertible() const { return true; }

inline Legion::LogicalPartition NoPartition::construct(Legion::LogicalRegion /*region*/,
                                                       bool /*complete*/) const
{
  return Legion::LogicalPartition::NO_PART;
}

inline bool NoPartition::has_launch_domain() const { return false; }

}  // namespace legate::detail
