/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_arrays/struct_physical_array.h>

namespace legate::detail {

inline StructPhysicalArray::StructPhysicalArray(
  InternalSharedPtr<Type> type,
  std::optional<InternalSharedPtr<PhysicalStore>> null_mask,
  SmallVector<InternalSharedPtr<PhysicalArray>>&& fields)
  : type_{std::move(type)}, null_mask_{std::move(null_mask)}, fields_{std::move(fields)}
{
}

inline std::int32_t StructPhysicalArray::dim() const { return fields_.front()->dim(); }

inline const InternalSharedPtr<Type>& StructPhysicalArray::type() const { return type_; }

inline bool StructPhysicalArray::nullable() const
{
  // Don't use null_mask() here, otherwise infinite recursive loop
  return null_mask_.has_value();
}

inline bool StructPhysicalArray::nested() const { return true; }

}  // namespace legate::detail
