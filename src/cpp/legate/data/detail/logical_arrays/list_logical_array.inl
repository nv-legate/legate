/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_arrays/base_logical_array.h>
#include <legate/data/detail/logical_arrays/list_logical_array.h>

namespace legate::detail {

inline std::uint32_t ListLogicalArray::dim() const { return descriptor_->dim(); }

inline const InternalSharedPtr<Type>& ListLogicalArray::type() const { return type_; }

inline const InternalSharedPtr<Shape>& ListLogicalArray::shape() const
{
  return descriptor_->shape();
}

inline std::size_t ListLogicalArray::volume() const { return descriptor_->volume(); }

inline bool ListLogicalArray::nullable() const { return descriptor_->nullable(); }

inline bool ListLogicalArray::nested() const { return true; }

inline std::uint32_t ListLogicalArray::num_children() const { return 2; }

inline bool ListLogicalArray::is_mapped() const
{
  return descriptor()->is_mapped() || vardata()->is_mapped();
}

inline const InternalSharedPtr<LogicalStore>& ListLogicalArray::null_mask() const
{
  return descriptor_->null_mask();
}

inline const InternalSharedPtr<LogicalStore>& ListLogicalArray::primary_store() const
{
  return descriptor_->primary_store();
}

}  // namespace legate::detail
