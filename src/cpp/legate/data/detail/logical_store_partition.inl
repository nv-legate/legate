/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_store_partition.h>
#include <legate/data/detail/transform.h>

namespace legate::detail {

inline LogicalStorePartition::LogicalStorePartition(
  InternalSharedPtr<Partition> partition,
  InternalSharedPtr<StoragePartition> storage_partition,
  InternalSharedPtr<LogicalStore> store)
  : partition_{std::move(partition)},
    storage_partition_{std::move(storage_partition)},
    store_{std::move(store)}
{
}

inline const InternalSharedPtr<Partition>& LogicalStorePartition::partition() const
{
  return partition_;
}

inline const InternalSharedPtr<StoragePartition>& LogicalStorePartition::storage_partition() const
{
  return storage_partition_;
}

inline const InternalSharedPtr<LogicalStore>& LogicalStorePartition::store() const
{
  return store_;
}

}  // namespace legate::detail
