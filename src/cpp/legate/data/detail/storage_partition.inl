/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/shape.h>
#include <legate/data/detail/storage_partition.h>

#include <utility>

namespace legate::detail {

inline StoragePartition::StoragePartition(InternalSharedPtr<Storage> parent,
                                          InternalSharedPtr<Partition> partition,
                                          bool complete)
  : complete_{complete},
    level_{parent->level() + 1},
    parent_{std::move(parent)},
    partition_{std::move(partition)}
{
}

inline const InternalSharedPtr<Partition>& StoragePartition::partition() const
{
  return partition_;
}

inline std::int32_t StoragePartition::level() const { return level_; }

}  // namespace legate::detail
