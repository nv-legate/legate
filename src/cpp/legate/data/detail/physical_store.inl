/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_store.h>

#include <utility>

namespace legate::detail {

inline PhysicalStore::PhysicalStore(std::int32_t dim,
                                    InternalSharedPtr<Type> type,
                                    GlobalRedopID redop_id,
                                    InternalSharedPtr<detail::TransformStack> transform,
                                    bool readable,
                                    bool writable,
                                    bool reducible)
  : transform_{std::move(transform)},
    type_{std::move(type)},
    dim_{dim},
    redop_id_{redop_id},
    readable_{readable},
    writable_{writable},
    reducible_{reducible}
{
}

inline std::int32_t PhysicalStore::dim() const { return dim_; }

inline const InternalSharedPtr<Type>& PhysicalStore::type() const { return type_; }

inline bool PhysicalStore::is_readable() const { return readable_; }

inline bool PhysicalStore::is_writable() const { return writable_; }

inline bool PhysicalStore::is_reducible() const { return reducible_; }

inline GlobalRedopID PhysicalStore::get_redop_id() const { return redop_id_; }

}  // namespace legate::detail
