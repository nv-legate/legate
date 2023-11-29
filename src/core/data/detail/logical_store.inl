/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/data/detail/logical_store.h"

namespace legate::detail {

inline uint64_t Storage::id() const { return storage_id_; }

inline bool Storage::unbound() const { return unbound_; }

inline size_t Storage::volume() const { return extents().volume(); }

inline int32_t Storage::dim() const { return dim_; }

inline std::shared_ptr<Type> Storage::type() const { return type_; }

inline Storage::Kind Storage::kind() const { return kind_; }

inline int32_t Storage::level() const { return level_; }

// ==========================================================================================

inline StoragePartition::StoragePartition(std::shared_ptr<Storage> parent,
                                          std::shared_ptr<Partition> partition,
                                          bool complete)
  : complete_{complete},
    level_{parent->level() + 1},
    parent_{std::move(parent)},
    partition_{std::move(partition)}
{
}

inline std::shared_ptr<Partition> StoragePartition::partition() const { return partition_; }

inline int32_t StoragePartition::level() const { return level_; }

// ==========================================================================================

inline size_t LogicalStore::volume() const { return extents().volume(); }

inline const std::shared_ptr<TransformStack>& LogicalStore::transform() const { return transform_; }

inline uint64_t LogicalStore::id() const { return store_id_; }

inline std::shared_ptr<Partition> LogicalStore::get_current_key_partition() const
{
  return key_partition_;
}

// ==========================================================================================

inline LogicalStorePartition::LogicalStorePartition(
  std::shared_ptr<Partition> partition,
  std::shared_ptr<StoragePartition> storage_partition,
  std::shared_ptr<LogicalStore> store)
  : partition_{std::move(partition)},
    storage_partition_{std::move(storage_partition)},
    store_{std::move(store)}
{
}

inline std::shared_ptr<Partition> LogicalStorePartition::partition() const { return partition_; }

inline std::shared_ptr<StoragePartition> LogicalStorePartition::storage_partition() const
{
  return storage_partition_;
}

inline std::shared_ptr<LogicalStore> LogicalStorePartition::store() const { return store_; }

}  // namespace legate::detail
