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

#include "core/mapping/detail/array.h"

namespace legate::mapping::detail {

inline std::vector<InternalSharedPtr<Store>> Array::stores() const
{
  std::vector<InternalSharedPtr<Store>> result;
  _stores(result);
  return result;
}

inline BaseArray::BaseArray(InternalSharedPtr<Store> data, InternalSharedPtr<Store> null_mask)
  : data_{std::move(data)}, null_mask_{std::move(null_mask)}
{
}

// ==========================================================================================

inline int32_t BaseArray::dim() const { return data_->dim(); }

inline legate::detail::ArrayKind BaseArray::kind() const { return legate::detail::ArrayKind::BASE; }

inline InternalSharedPtr<legate::detail::Type> BaseArray::type() const { return data_->type(); }

inline bool BaseArray::nullable() const { return null_mask_ != nullptr; }

inline bool BaseArray::nested() const { return false; }

inline InternalSharedPtr<Store> BaseArray::data() const { return data_; }

// ==========================================================================================

inline ListArray::ListArray(InternalSharedPtr<legate::detail::Type> type,
                            InternalSharedPtr<BaseArray> descriptor,
                            InternalSharedPtr<Array> vardata)
  : type_{std::move(type)}, descriptor_{std::move(descriptor)}, vardata_{std::move(vardata)}
{
}

inline legate::detail::ArrayKind ListArray::kind() const { return legate::detail::ArrayKind::LIST; }

inline InternalSharedPtr<legate::detail::Type> ListArray::type() const { return type_; }

inline bool ListArray::nullable() const { return descriptor_->nullable(); }

inline bool ListArray::nested() const { return true; }

inline InternalSharedPtr<Store> ListArray::null_mask() const { return descriptor_->null_mask(); }

inline InternalSharedPtr<Array> ListArray::descriptor() const { return descriptor_; }

inline InternalSharedPtr<Array> ListArray::vardata() const { return vardata_; }

// ==========================================================================================

inline StructArray::StructArray(InternalSharedPtr<legate::detail::Type> type,
                                InternalSharedPtr<Store> null_mask,
                                std::vector<InternalSharedPtr<Array>>&& fields)
  : type_{std::move(type)}, null_mask_{std::move(null_mask)}, fields_{std::move(fields)}
{
}

inline legate::detail::ArrayKind StructArray::kind() const
{
  return legate::detail::ArrayKind::STRUCT;
}

inline InternalSharedPtr<legate::detail::Type> StructArray::type() const { return type_; }

inline bool StructArray::nullable() const { return null_mask_ != nullptr; }

inline bool StructArray::nested() const { return true; }

}  // namespace legate::mapping::detail
