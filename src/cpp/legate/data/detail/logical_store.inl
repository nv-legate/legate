/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/shape.h>
#include <legate/data/detail/transform.h>

#include <utility>

namespace legate::detail {

inline std::uint64_t Storage::id() const { return storage_id_; }

inline bool Storage::replicated() const { return replicated_; }

inline bool Storage::unbound() const { return unbound_; }

inline const InternalSharedPtr<Shape>& Storage::shape() const { return shape_; }

inline Span<const std::uint64_t> Storage::extents() const { return shape()->extents(); }

inline std::size_t Storage::volume() const { return shape()->volume(); }

inline std::uint32_t Storage::dim() const { return shape()->ndim(); }

inline Storage::Kind Storage::kind() const { return kind_; }

inline std::int32_t Storage::level() const { return level_; }

inline std::size_t Storage::scalar_offset() const { return scalar_offset_; }

inline std::string_view Storage::provenance() const { return provenance_; }

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

inline const InternalSharedPtr<Partition>& StoragePartition::partition() const
{
  return partition_;
}

inline std::int32_t StoragePartition::level() const { return level_; }

// ==========================================================================================

inline bool LogicalStore::unbound() const { return get_storage()->unbound(); }

inline const InternalSharedPtr<Shape>& LogicalStore::shape() const { return shape_; }

inline Span<const std::uint64_t> LogicalStore::extents() const { return shape()->extents(); }

inline std::size_t LogicalStore::volume() const { return shape()->volume(); }

inline std::size_t LogicalStore::storage_size() const
{
  return get_storage()->volume() * type()->size();
}

inline std::uint32_t LogicalStore::dim() const { return shape()->ndim(); }

inline const InternalSharedPtr<TransformStack>& LogicalStore::transform() const
{
  return transform_;
}

inline bool LogicalStore::overlaps(const InternalSharedPtr<LogicalStore>& other) const
{
  return get_storage()->overlaps(other->storage_);
}

inline bool LogicalStore::has_scalar_storage() const
{
  return get_storage()->kind() != Storage::Kind::REGION_FIELD;
}

inline const InternalSharedPtr<Type>& LogicalStore::type() const { return type_; }

inline bool LogicalStore::transformed() const { return !transform_->identity(); }

inline std::uint64_t LogicalStore::id() const { return store_id_; }

inline const InternalSharedPtr<Storage>& LogicalStore::get_storage() const { return storage_; }

inline const InternalSharedPtr<LogicalRegionField>& LogicalStore::get_region_field() const
{
  return storage_->get_region_field();
}

inline Legion::Future LogicalStore::get_future() const { return get_storage()->get_future(); }

inline Legion::FutureMap LogicalStore::get_future_map() const
{
  return get_storage()->get_future_map();
}

// We can't rely on the mapped_ here, as this store might just be an alias to some other store the
// user created a physical store with. The storage_ holds the ground truth
inline bool LogicalStore::is_mapped() const { return get_storage()->is_mapped(); }

inline bool LogicalStore::needs_flush() const { return unbound() || is_mapped(); }

inline const InternalSharedPtr<Partition>& LogicalStore::get_current_key_partition() const
{
  return key_partition_.value();  // NOLINT(bugprone-unchecked-optional-access)
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

inline InternalSharedPtr<StoragePartition> create_storage_partition(
  const InternalSharedPtr<Storage>& self,
  InternalSharedPtr<Partition> partition,
  std::optional<bool> complete)
{
  return self->create_partition(self, std::move(partition), std::move(complete));
}

inline InternalSharedPtr<Storage> slice_storage(
  const InternalSharedPtr<Storage>& self,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
  SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets)
{
  return self->slice(self, std::move(tile_shape), std::move(offsets));
}

inline InternalSharedPtr<LogicalStore> slice_store(const InternalSharedPtr<LogicalStore>& self,
                                                   std::int32_t dim,
                                                   Slice sl)
{
  return self->slice_(self, dim, std::move(sl));
}

inline InternalSharedPtr<LogicalStorePartition> partition_store_by_tiling(
  const InternalSharedPtr<LogicalStore>& self,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
  std::optional<SmallVector<std::uint64_t, LEGATE_MAX_DIM>> color_shape)
{
  return self->partition_by_tiling_(self, std::move(tile_shape), std::move(color_shape));
}

inline InternalSharedPtr<LogicalStorePartition> create_store_partition(
  const InternalSharedPtr<LogicalStore>& self,
  InternalSharedPtr<Partition> partition,
  std::optional<bool> complete)
{
  return self->create_partition_(self, std::move(partition), std::move(complete));
}

inline StoreAnalyzable store_to_launcher_arg(const InternalSharedPtr<LogicalStore>& self,
                                             const Variable* variable,
                                             const Strategy& strategy,
                                             const Domain& launch_domain,
                                             const std::optional<SymbolicPoint>& projection,
                                             Legion::PrivilegeMode privilege,
                                             GlobalRedopID redop)
{
  return self->to_launcher_arg_(
    self, variable, strategy, launch_domain, projection, privilege, redop);
}

inline RegionFieldArg store_to_launcher_arg_for_fixup(const InternalSharedPtr<LogicalStore>& self,
                                                      const Domain& launch_domain,
                                                      Legion::PrivilegeMode privilege)
{
  return self->to_launcher_arg_for_fixup_(self, launch_domain, privilege);
}

}  // namespace legate::detail
