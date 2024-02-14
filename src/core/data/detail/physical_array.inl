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

#include "core/data/detail/physical_array.h"

namespace legate::detail {

inline std::vector<InternalSharedPtr<PhysicalStore>> PhysicalArray::stores() const
{
  std::vector<InternalSharedPtr<PhysicalStore>> result;
  _stores(result);
  return result;
}

// ==========================================================================================

inline BasePhysicalArray::BasePhysicalArray(InternalSharedPtr<PhysicalStore> data,
                                            InternalSharedPtr<PhysicalStore> null_mask)
  : data_{std::move(data)}, null_mask_{std::move(null_mask)}
{
}

inline std::int32_t BasePhysicalArray::dim() const { return data_->dim(); }

inline ArrayKind BasePhysicalArray::kind() const { return ArrayKind::BASE; }

inline InternalSharedPtr<Type> BasePhysicalArray::type() const { return data_->type(); }

inline bool BasePhysicalArray::nullable() const { return null_mask_ != nullptr; }

inline bool BasePhysicalArray::nested() const { return false; }

inline bool BasePhysicalArray::valid() const { return data_->valid(); }

inline InternalSharedPtr<PhysicalStore> BasePhysicalArray::data() const { return data_; }

inline Domain BasePhysicalArray::domain() const { return data_->domain(); }

// ==========================================================================================

inline ListPhysicalArray::ListPhysicalArray(InternalSharedPtr<Type> type,
                                            InternalSharedPtr<BasePhysicalArray> descriptor,
                                            InternalSharedPtr<PhysicalArray> vardata)
  : type_{std::move(type)}, descriptor_{std::move(descriptor)}, vardata_{std::move(vardata)}
{
}

inline std::int32_t ListPhysicalArray::dim() const { return descriptor_->dim(); }

inline ArrayKind ListPhysicalArray::kind() const { return ArrayKind::LIST; }

inline InternalSharedPtr<Type> ListPhysicalArray::type() const { return type_; }

inline bool ListPhysicalArray::unbound() const
{
  return descriptor_->unbound() || vardata_->unbound();
}

inline bool ListPhysicalArray::nullable() const { return descriptor_->nullable(); }

inline bool ListPhysicalArray::nested() const { return true; }

inline InternalSharedPtr<PhysicalStore> ListPhysicalArray::null_mask() const
{
  return descriptor_->null_mask();
}

inline InternalSharedPtr<PhysicalArray> ListPhysicalArray::descriptor() const
{
  return descriptor_;
}

inline InternalSharedPtr<PhysicalArray> ListPhysicalArray::vardata() const { return vardata_; }

inline Domain ListPhysicalArray::domain() const { return descriptor_->domain(); }

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

inline InternalSharedPtr<Type> StructPhysicalArray::type() const { return type_; }

inline bool StructPhysicalArray::nullable() const { return null_mask_ != nullptr; }

inline bool StructPhysicalArray::nested() const { return true; }

}  // namespace legate::detail
