/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/array.h>

namespace legate::mapping::detail {

inline BaseArray::BaseArray(InternalSharedPtr<Store> data,
                            std::optional<InternalSharedPtr<Store>> null_mask)
  : data_{std::move(data)}, null_mask_{std::move(null_mask)}
{
}

// ==========================================================================================

inline std::int32_t BaseArray::dim() const { return data()->dim(); }

inline legate::detail::ArrayKind BaseArray::kind() const { return legate::detail::ArrayKind::BASE; }

inline const InternalSharedPtr<legate::detail::Type>& BaseArray::type() const
{
  return data()->type();
}

inline bool BaseArray::nullable() const { return null_mask_.has_value(); }

inline bool BaseArray::nested() const { return false; }

inline bool BaseArray::valid() const { return data()->valid(); }

inline const InternalSharedPtr<Store>& BaseArray::data() const { return data_; }

// ==========================================================================================

inline ListArray::ListArray(InternalSharedPtr<legate::detail::Type> type,
                            InternalSharedPtr<BaseArray> descriptor,
                            InternalSharedPtr<Array> vardata)
  : type_{std::move(type)}, descriptor_{std::move(descriptor)}, vardata_{std::move(vardata)}
{
}

inline legate::detail::ArrayKind ListArray::kind() const { return legate::detail::ArrayKind::LIST; }

inline const InternalSharedPtr<legate::detail::Type>& ListArray::type() const { return type_; }

inline bool ListArray::nullable() const { return descriptor()->nullable(); }

inline bool ListArray::nested() const { return true; }

inline const InternalSharedPtr<Store>& ListArray::null_mask() const
{
  return descriptor()->null_mask();
}

inline const InternalSharedPtr<BaseArray>& ListArray::descriptor() const { return descriptor_; }

inline const InternalSharedPtr<Array>& ListArray::vardata() const { return vardata_; }

// ==========================================================================================

inline StructArray::StructArray(InternalSharedPtr<legate::detail::Type> type,
                                std::optional<InternalSharedPtr<Store>> null_mask,
                                legate::detail::SmallVector<InternalSharedPtr<Array>>&& fields)
  : type_{std::move(type)}, null_mask_{std::move(null_mask)}, fields_{std::move(fields)}
{
}

inline legate::detail::ArrayKind StructArray::kind() const
{
  return legate::detail::ArrayKind::STRUCT;
}

inline const InternalSharedPtr<legate::detail::Type>& StructArray::type() const { return type_; }

inline bool StructArray::nullable() const { return null_mask_.has_value(); }

inline bool StructArray::nested() const { return true; }

inline Span<const InternalSharedPtr<Array>> StructArray::fields() const { return fields_; }

}  // namespace legate::mapping::detail
