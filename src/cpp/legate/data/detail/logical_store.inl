/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

inline const Point& LogicalStore::extents() const { return shape()->extents(); }

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

inline const InternalSharedPtr<Type>& LogicalStore::type() const { return type_; }

inline bool LogicalStore::transformed() const { return !transform_->identity(); }

inline std::uint64_t LogicalStore::id() const { return store_id_; }

inline const InternalSharedPtr<Storage>& LogicalStore::get_storage() const { return storage_; }

inline void LogicalStore::finalize_unbound() { get_storage()->finalize_unbound(); }

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

}  // namespace legate::detail
