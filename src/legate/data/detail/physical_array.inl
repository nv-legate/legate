/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "legate/data/detail/physical_array.h"

namespace legate::detail {

inline BasePhysicalArray::BasePhysicalArray(InternalSharedPtr<PhysicalStore> data,
                                            InternalSharedPtr<PhysicalStore> null_mask)
  : data_{std::move(data)}, null_mask_{std::move(null_mask)}
{
}

inline std::int32_t BasePhysicalArray::dim() const { return data()->dim(); }

inline ArrayKind BasePhysicalArray::kind() const { return ArrayKind::BASE; }

inline const InternalSharedPtr<Type>& BasePhysicalArray::type() const { return data()->type(); }

inline bool BasePhysicalArray::nullable() const
{
  // Don't use null_mask() here, otherwise infinite recursive loop
  return null_mask_ != nullptr;
}

inline bool BasePhysicalArray::nested() const { return false; }

inline bool BasePhysicalArray::valid() const { return data()->valid(); }

inline const InternalSharedPtr<PhysicalStore>& BasePhysicalArray::data() const { return data_; }

inline Domain BasePhysicalArray::domain() const { return data()->domain(); }

// ==========================================================================================

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

// ==========================================================================================

inline StructPhysicalArray::StructPhysicalArray(
  InternalSharedPtr<Type> type,
  InternalSharedPtr<PhysicalStore> null_mask,
  std::vector<InternalSharedPtr<PhysicalArray>>&& fields)
  : type_{std::move(type)}, null_mask_{std::move(null_mask)}, fields_{std::move(fields)}
{
}

inline std::int32_t StructPhysicalArray::dim() const { return fields_.front()->dim(); }

inline ArrayKind StructPhysicalArray::kind() const { return ArrayKind::STRUCT; }

inline const InternalSharedPtr<Type>& StructPhysicalArray::type() const { return type_; }

inline bool StructPhysicalArray::nullable() const
{
  // Don't use null_mask() here, otherwise infinite recursive loop
  return null_mask_ != nullptr;
}

inline bool StructPhysicalArray::nested() const { return true; }

}  // namespace legate::detail
