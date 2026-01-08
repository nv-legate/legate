/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_arrays/base_logical_array.h>

namespace legate::detail {

inline BaseLogicalArray::BaseLogicalArray(InternalSharedPtr<LogicalStore> data,
                                          std::optional<InternalSharedPtr<LogicalStore>> null_mask)
  : data_{std::move(data)}, null_mask_{std::move(null_mask)}
{
  auto&& code = type()->code;

  LEGATE_ASSERT(code != Type::Code::STRING && code != Type::Code::LIST);
}

inline std::uint32_t BaseLogicalArray::dim() const { return data_->dim(); }

inline const InternalSharedPtr<Type>& BaseLogicalArray::type() const { return data_->type(); }

inline const InternalSharedPtr<Shape>& BaseLogicalArray::shape() const { return data_->shape(); }

inline std::size_t BaseLogicalArray::volume() const { return data_->volume(); }

inline bool BaseLogicalArray::nullable() const { return null_mask_.has_value(); }

inline bool BaseLogicalArray::nested() const { return false; }

inline std::uint32_t BaseLogicalArray::num_children() const { return 0; }

inline bool BaseLogicalArray::is_mapped() const
{
  return data()->is_mapped() || (nullable() && null_mask()->is_mapped());
}

inline const InternalSharedPtr<LogicalStore>& BaseLogicalArray::data() const { return data_; }

inline const InternalSharedPtr<LogicalStore>& BaseLogicalArray::primary_store() const
{
  return data();
}

}  // namespace legate::detail
