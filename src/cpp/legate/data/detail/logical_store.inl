/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/shape.h>
#include <legate/data/detail/storage.h>
#include <legate/data/detail/transform/transform_stack.h>

#include <utility>

namespace legate::detail {

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

inline const std::optional<InternalSharedPtr<Partition>>& LogicalStore::get_current_key_partition()
  const
{
  return key_partition_;
}

inline InternalSharedPtr<LogicalStore> slice_store(const InternalSharedPtr<LogicalStore>& self,
                                                   std::int32_t dim,
                                                   Slice sl)
{
  return self->slice_(self, dim, std::move(sl));
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
