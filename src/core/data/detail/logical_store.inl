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

inline InternalSharedPtr<Type> Storage::type() const { return type_; }

inline Storage::Kind Storage::kind() const { return kind_; }

inline int32_t Storage::level() const { return level_; }

// ==========================================================================================

inline StoragePartition::StoragePartition(InternalSharedPtr<Storage> parent,
                                          InternalSharedPtr<Partition> partition,
                                          bool complete)
  : complete_{complete},
    level_{parent->level() + 1},
    parent_{std::move(parent)},
    partition_{std::move(partition)}
{
}

inline InternalSharedPtr<Partition> StoragePartition::partition() const { return partition_; }

inline int32_t StoragePartition::level() const { return level_; }

// ==========================================================================================

inline size_t LogicalStore::volume() const { return extents().volume(); }

inline const InternalSharedPtr<TransformStack>& LogicalStore::transform() const
{
  return transform_;
}

inline uint64_t LogicalStore::id() const { return store_id_; }

inline InternalSharedPtr<Partition> LogicalStore::get_current_key_partition() const
{
  return key_partition_;
}

// ==========================================================================================

inline LogicalStorePartition::LogicalStorePartition(
  InternalSharedPtr<Partition> partition,
  InternalSharedPtr<StoragePartition> storage_partition,
  InternalSharedPtr<LogicalStore> store)
  : partition_{std::move(partition)},
    storage_partition_{std::move(storage_partition)},
    store_{std::move(store)}
{
}

inline InternalSharedPtr<Partition> LogicalStorePartition::partition() const { return partition_; }

inline InternalSharedPtr<StoragePartition> LogicalStorePartition::storage_partition() const
{
  return storage_partition_;
}

inline InternalSharedPtr<LogicalStore> LogicalStorePartition::store() const { return store_; }

inline InternalSharedPtr<LogicalStore> slice_store(const InternalSharedPtr<LogicalStore>& self,
                                                   int32_t dim,
                                                   Slice sl)
{
  return self->slice(self, dim, sl);
}

inline InternalSharedPtr<LogicalStorePartition> partition_store_by_tiling(
  const InternalSharedPtr<LogicalStore>& self, Shape tile_shape)
{
  return self->partition_by_tiling(self, std::move(tile_shape));
}

inline InternalSharedPtr<LogicalStorePartition> create_store_partition(
  const InternalSharedPtr<LogicalStore>& self,
  InternalSharedPtr<Partition> partition,
  std::optional<bool> complete)
{
  return self->create_partition(self, std::move(partition), std::move(complete));
}

inline std::unique_ptr<Analyzable> store_to_launcher_arg(
  const InternalSharedPtr<LogicalStore>& self,
  const Variable* variable,
  const Strategy& strategy,
  const Domain& launch_domain,
  const std::optional<SymbolicPoint>& projection,
  Legion::PrivilegeMode privilege,
  int64_t redop)
{
  return self->to_launcher_arg(
    self, variable, strategy, launch_domain, projection, privilege, redop);
}

inline std::unique_ptr<Analyzable> store_to_launcher_arg_for_fixup(
  const InternalSharedPtr<LogicalStore>& self,
  const Domain& launch_domain,
  Legion::PrivilegeMode privilege)
{
  return self->to_launcher_arg_for_fixup(self, launch_domain, privilege);
}

}  // namespace legate::detail
