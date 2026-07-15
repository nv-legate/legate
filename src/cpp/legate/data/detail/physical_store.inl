/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_store.h>
#include <legate/utilities/detail/legion_utilities.h>

#include <utility>

namespace legate::detail {

inline PhysicalStore::PhysicalStore(std::int32_t dim,
                                    InternalSharedPtr<Type> type,
                                    InternalSharedPtr<detail::TransformStack> transform,
                                    Legion::PrivilegeMode priv,
                                    InternalSharedPtr<InlineStorage> storage,
                                    const Domain& domain)
  : transform_{std::move(transform)},
    type_{std::move(type)},
    dim_{dim},
    readable_{has_privilege(priv, LEGION_READ_PRIV)},
    writable_{has_privilege(priv, LEGION_WRITE_PRIV)},
    reducible_{has_privilege(priv, LEGION_REDUCE_PRIV)},
    inline_storage_{std::move(storage)},
    domain_{domain}
{
}

inline bool PhysicalStore::valid() const { return true; }

inline bool PhysicalStore::transformed() const { return !transform_->identity(); }

inline std::int32_t PhysicalStore::dim() const { return dim_; }

inline const InternalSharedPtr<Type>& PhysicalStore::type() const { return type_; }

inline mapping::StoreTarget PhysicalStore::target() const { return storage_()->target(); }

inline bool PhysicalStore::is_readable() const { return readable_; }

inline bool PhysicalStore::is_writable() const { return writable_; }

inline bool PhysicalStore::is_reducible() const { return reducible_; }

inline const InternalSharedPtr<InlineStorage>& PhysicalStore::storage_() const
{
  return inline_storage_;
}

}  // namespace legate::detail
