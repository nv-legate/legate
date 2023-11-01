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

#include "core/data/detail/logical_array.h"

namespace legate::detail {

inline BaseLogicalArray::BaseLogicalArray(std::shared_ptr<LogicalStore> data,
                                          std::shared_ptr<LogicalStore> null_mask)
  : data_{std::move(data)}, null_mask_{std::move(null_mask)}
{
  assert(data_ != nullptr);
}

inline int32_t BaseLogicalArray::dim() const { return data_->dim(); }

inline ArrayKind BaseLogicalArray::kind() const { return ArrayKind::BASE; }

inline std::shared_ptr<Type> BaseLogicalArray::type() const { return data_->type(); }

inline const Shape& BaseLogicalArray::extents() const { return data_->extents(); }

inline size_t BaseLogicalArray::volume() const { return data_->volume(); }

inline bool BaseLogicalArray::nullable() const { return null_mask_ != nullptr; }

inline bool BaseLogicalArray::nested() const { return false; }

inline uint32_t BaseLogicalArray::num_children() const { return 0; }

inline std::shared_ptr<LogicalStore> BaseLogicalArray::data() const { return data_; }

inline std::shared_ptr<LogicalStore> BaseLogicalArray::primary_store() const { return data(); }

// ==========================================================================================

inline ListLogicalArray::ListLogicalArray(std::shared_ptr<Type> type,
                                          std::shared_ptr<BaseLogicalArray> descriptor,
                                          std::shared_ptr<LogicalArray> vardata)
  : type_{std::move(type)}, descriptor_{std::move(descriptor)}, vardata_{std::move(vardata)}
{
}

inline int32_t ListLogicalArray::dim() const { return descriptor_->dim(); }

inline ArrayKind ListLogicalArray::kind() const { return ArrayKind::LIST; }

inline std::shared_ptr<Type> ListLogicalArray::type() const { return type_; }

inline const Shape& ListLogicalArray::extents() const { return descriptor_->extents(); }

inline size_t ListLogicalArray::volume() const { return descriptor_->volume(); }

inline bool ListLogicalArray::nullable() const { return descriptor_->nullable(); }

inline bool ListLogicalArray::nested() const { return true; }

inline uint32_t ListLogicalArray::num_children() const { return 2; }

inline std::shared_ptr<LogicalStore> ListLogicalArray::null_mask() const
{
  return descriptor_->null_mask();
}

inline std::shared_ptr<LogicalStore> ListLogicalArray::primary_store() const
{
  return descriptor_->primary_store();
}

// ==========================================================================================

inline StructLogicalArray::StructLogicalArray(std::shared_ptr<Type> type,
                                              std::shared_ptr<LogicalStore> null_mask,
                                              std::vector<std::shared_ptr<LogicalArray>>&& fields)
  : type_{std::move(type)}, null_mask_{std::move(null_mask)}, fields_{std::move(fields)}
{
}

inline ArrayKind StructLogicalArray::kind() const { return ArrayKind::STRUCT; }

inline std::shared_ptr<Type> StructLogicalArray::type() const { return type_; }

inline bool StructLogicalArray::nullable() const { return null_mask_ != nullptr; }

inline bool StructLogicalArray::nested() const { return true; }

inline uint32_t StructLogicalArray::num_children() const
{
  return static_cast<uint32_t>(fields_.size());
}

}  // namespace legate::detail
