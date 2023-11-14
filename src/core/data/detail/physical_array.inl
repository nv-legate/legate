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

inline std::vector<std::shared_ptr<PhysicalStore>> PhysicalArray::stores() const
{
  std::vector<std::shared_ptr<PhysicalStore>> result;
  _stores(result);
  return result;
}

// ==========================================================================================

inline BasePhysicalArray::BasePhysicalArray(std::shared_ptr<PhysicalStore> data,
                                            std::shared_ptr<PhysicalStore> null_mask)
  : data_{std::move(data)}, null_mask_{std::move(null_mask)}
{
}

inline int32_t BasePhysicalArray::dim() const { return data_->dim(); }

inline ArrayKind BasePhysicalArray::kind() const { return ArrayKind::BASE; }

inline std::shared_ptr<Type> BasePhysicalArray::type() const { return data_->type(); }

inline bool BasePhysicalArray::nullable() const { return null_mask_ != nullptr; }

inline bool BasePhysicalArray::nested() const { return false; }

inline bool BasePhysicalArray::valid() const { return data_->valid(); }

inline std::shared_ptr<PhysicalStore> BasePhysicalArray::data() const { return data_; }

inline Domain BasePhysicalArray::domain() const { return data_->domain(); }

// ==========================================================================================

inline ListPhysicalArray::ListPhysicalArray(std::shared_ptr<Type> type,
                                            std::shared_ptr<BasePhysicalArray> descriptor,
                                            std::shared_ptr<PhysicalArray> vardata)
  : type_{std::move(type)}, descriptor_{std::move(descriptor)}, vardata_{std::move(vardata)}
{
}

inline int32_t ListPhysicalArray::dim() const { return descriptor_->dim(); }

inline ArrayKind ListPhysicalArray::kind() const { return ArrayKind::LIST; }

inline std::shared_ptr<Type> ListPhysicalArray::type() const { return type_; }

inline bool ListPhysicalArray::unbound() const
{
  return descriptor_->unbound() || vardata_->unbound();
}

inline bool ListPhysicalArray::nullable() const { return vardata_->nullable(); }

inline bool ListPhysicalArray::nested() const { return true; }

inline std::shared_ptr<PhysicalStore> ListPhysicalArray::null_mask() const
{
  return descriptor_->null_mask();
}

inline std::shared_ptr<PhysicalArray> ListPhysicalArray::descriptor() const { return descriptor_; }

inline std::shared_ptr<PhysicalArray> ListPhysicalArray::vardata() const { return vardata_; }

inline Domain ListPhysicalArray::domain() const { return descriptor_->domain(); }

// ==========================================================================================

inline StructPhysicalArray::StructPhysicalArray(
  std::shared_ptr<Type> type,
  std::shared_ptr<PhysicalStore> null_mask,
  std::vector<std::shared_ptr<PhysicalArray>>&& fields)
  : type_{std::move(type)}, null_mask_{std::move(null_mask)}, fields_{std::move(fields)}
{
}

inline int32_t StructPhysicalArray::dim() const { return fields_.front()->dim(); }

inline ArrayKind StructPhysicalArray::kind() const { return ArrayKind::STRUCT; }

inline std::shared_ptr<Type> StructPhysicalArray::type() const { return type_; }

inline bool StructPhysicalArray::nullable() const { return null_mask_ != nullptr; }

inline bool StructPhysicalArray::nested() const { return true; }

}  // namespace legate::detail
