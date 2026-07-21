/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_store.h>

#include <utility>

namespace legate::detail {

inline PhysicalStore::PhysicalStore(InternalSharedPtr<Type> type,
                                    StorePermission permission,
                                    InternalSharedPtr<InlineStorage> storage,
                                    PhysicalStoreLayout layout)
  : type_{std::move(type)},
    dim_{static_cast<std::int32_t>(layout.domain.dim())},
    readable_{permission == StorePermission::READ_WRITE},
    writable_{permission == StorePermission::READ_WRITE},
    reducible_{permission == StorePermission::REDUCE},
    inline_storage_{std::move(storage)},
    base_{layout.base},
    strides_{std::move(layout.strides)},
    domain_{std::move(layout.domain)}
{
}

inline PhysicalStore::PhysicalStore(std::int32_t dim,
                                    InternalSharedPtr<Type> type,
                                    mapping::StoreTarget target,
                                    InternalSharedPtr<LogicalStore> originating_store)
  : type_{std::move(type)},
    dim_{dim},
    writable_{true},
    unbound_{true},
    unbound_target_{target},
    originating_store_{std::move(originating_store)}
{
}

inline bool PhysicalStore::valid() const { return true; }

inline std::int32_t PhysicalStore::dim() const { return dim_; }

inline const InternalSharedPtr<Type>& PhysicalStore::type() const { return type_; }

inline const Domain& PhysicalStore::domain() const { return domain_; }

inline void* PhysicalStore::base() const { return base_; }

inline const DomainPoint& PhysicalStore::strides() const { return strides_; }

inline mapping::StoreTarget PhysicalStore::target() const
{
  return unbound_ ? unbound_target_ : storage_()->target();
}

inline bool PhysicalStore::is_readable() const { return readable_; }

inline bool PhysicalStore::is_writable() const { return writable_; }

inline bool PhysicalStore::is_reducible() const { return reducible_; }

inline bool PhysicalStore::is_unbound_store() const { return unbound_; }

inline const InternalSharedPtr<InlineStorage>& PhysicalStore::storage_() const
{
  return inline_storage_;
}

}  // namespace legate::detail
