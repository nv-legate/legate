/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_arrays/struct_logical_array.h>

namespace legate::detail {

inline StructLogicalArray::StructLogicalArray(
  InternalSharedPtr<Type> type,
  std::optional<InternalSharedPtr<LogicalStore>> null_mask,
  SmallVector<InternalSharedPtr<LogicalArray>>&& fields)
  : type_{std::move(type)}, null_mask_{std::move(null_mask)}, fields_{std::move(fields)}
{
}

inline ArrayKind StructLogicalArray::kind() const { return ArrayKind::STRUCT; }

inline const InternalSharedPtr<Type>& StructLogicalArray::type() const { return type_; }

inline bool StructLogicalArray::nullable() const { return null_mask_.has_value(); }

inline bool StructLogicalArray::nested() const { return true; }

inline std::uint32_t StructLogicalArray::num_children() const
{
  return static_cast<std::uint32_t>(fields_.size());
}

inline Span<const InternalSharedPtr<LogicalArray>> StructLogicalArray::fields() const
{
  return fields_;
}

}  // namespace legate::detail
