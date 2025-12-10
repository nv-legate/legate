/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_arrays/struct_logical_array.h>

namespace legate::detail {

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
