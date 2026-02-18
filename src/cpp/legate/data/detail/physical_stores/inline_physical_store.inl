/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_stores/inline_physical_store.h>
#include <legate/utilities/detail/legion_utilities.h>

namespace legate::detail {

inline InlinePhysicalStore::InlinePhysicalStore(std::int32_t dim,
                                                InternalSharedPtr<Type> type,
                                                GlobalRedopID redop_id,
                                                InternalSharedPtr<detail::TransformStack> transform,
                                                Legion::PrivilegeMode priv,
                                                InternalSharedPtr<InlineStorage> storage,
                                                const Domain& domain)
  : PhysicalStore{dim,
                  std::move(type),
                  redop_id,
                  std::move(transform),
                  /* readable */ has_privilege(priv, LEGION_READ_PRIV),
                  /* writable */ has_privilege(priv, LEGION_WRITE_PRIV),
                  /* reducible */ has_privilege(priv, LEGION_REDUCE_PRIV)},
    inline_storage_{std::move(storage)},
    domain_{domain}
{
}

inline bool InlinePhysicalStore::valid() const { return true; }

inline bool InlinePhysicalStore::is_partitioned() const { return false; }

inline const InternalSharedPtr<InlineStorage>& InlinePhysicalStore::storage_() const
{
  return inline_storage_;
}

inline std::pair<Realm::RegionInstance, Realm::FieldID> InlinePhysicalStore::get_region_instance()
  const
{
  return storage_()->region_instance();
}

inline mapping::StoreTarget InlinePhysicalStore::target() const { return storage_()->target(); }

}  // namespace legate::detail
