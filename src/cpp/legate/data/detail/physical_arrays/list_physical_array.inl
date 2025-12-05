/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_arrays/base_physical_array.h>
#include <legate/data/detail/physical_arrays/list_physical_array.h>

namespace legate::detail {

inline ListPhysicalArray::ListPhysicalArray(InternalSharedPtr<Type> type,
                                            InternalSharedPtr<BasePhysicalArray> descriptor,
                                            InternalSharedPtr<PhysicalArray> vardata)
  : type_{std::move(type)}, descriptor_{std::move(descriptor)}, vardata_{std::move(vardata)}
{
}

inline std::int32_t ListPhysicalArray::dim() const { return descriptor()->dim(); }

inline ArrayKind ListPhysicalArray::kind() const { return ArrayKind::LIST; }

inline const InternalSharedPtr<Type>& ListPhysicalArray::type() const { return type_; }

inline bool ListPhysicalArray::unbound() const
{
  return descriptor()->unbound() || vardata()->unbound();
}

inline bool ListPhysicalArray::nullable() const { return descriptor()->nullable(); }

inline bool ListPhysicalArray::nested() const { return true; }

inline const InternalSharedPtr<PhysicalStore>& ListPhysicalArray::null_mask() const
{
  return descriptor()->null_mask();
}

inline const InternalSharedPtr<BasePhysicalArray>& ListPhysicalArray::descriptor() const
{
  return descriptor_;
}

inline const InternalSharedPtr<PhysicalArray>& ListPhysicalArray::vardata() const
{
  return vardata_;
}

inline Domain ListPhysicalArray::domain() const { return descriptor()->domain(); }

}  // namespace legate::detail
