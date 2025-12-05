/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/physical_arrays/list_physical_array.h>

#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <stdexcept>

namespace legate::detail {

bool ListPhysicalArray::valid() const
{
  const auto descr_valid = descriptor()->valid();

  LEGATE_ASSERT(descr_valid == vardata()->valid());
  return descr_valid;
}

InternalSharedPtr<PhysicalArray> ListPhysicalArray::child(std::uint32_t index) const
{
  switch (index) {
    case 0: return descriptor();
    case 1: return vardata();
    default: {  // legate-lint: no-switch-default
      throw TracedException<std::out_of_range>{
        fmt::format("List array does not have child {}", index)};
      break;
    }
  }
  return {};
}

void ListPhysicalArray::populate_stores(SmallVector<InternalSharedPtr<PhysicalStore>>& result) const
{
  descriptor()->populate_stores(result);
  vardata()->populate_stores(result);
}

void ListPhysicalArray::check_shape_dimension(std::int32_t dim) const
{
  descriptor()->check_shape_dimension(dim);
}

}  // namespace legate::detail
