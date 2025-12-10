/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_arrays/base_physical_array.h>
#include <legate/data/detail/physical_store.h>

namespace legate::detail {

inline BasePhysicalArray::BasePhysicalArray(
  InternalSharedPtr<PhysicalStore> data, std::optional<InternalSharedPtr<PhysicalStore>> null_mask)
  : data_{std::move(data)}, null_mask_{std::move(null_mask)}
{
}

inline std::int32_t BasePhysicalArray::dim() const { return data()->dim(); }

inline const InternalSharedPtr<Type>& BasePhysicalArray::type() const { return data()->type(); }

inline bool BasePhysicalArray::nullable() const
{
  // Don't use null_mask() here, otherwise infinite recursive loop
  return null_mask_.has_value();
}

inline bool BasePhysicalArray::nested() const { return false; }

inline bool BasePhysicalArray::valid() const { return data()->valid(); }

inline const InternalSharedPtr<PhysicalStore>& BasePhysicalArray::data() const { return data_; }

inline Domain BasePhysicalArray::domain() const { return data()->domain(); }

}  // namespace legate::detail
