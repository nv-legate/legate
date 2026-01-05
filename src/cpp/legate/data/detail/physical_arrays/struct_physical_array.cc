/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/physical_arrays/struct_physical_array.h>

#include <legate/data/detail/physical_store.h>
#include <legate/utilities/detail/traced_exception.h>

#include <algorithm>
#include <stdexcept>

namespace legate::detail {

bool StructPhysicalArray::unbound() const
{
  return std::any_of(fields_.begin(), fields_.end(), [](auto&& field) { return field->unbound(); });
}

bool StructPhysicalArray::valid() const
{
  const auto result =
    std::all_of(fields_.begin(), fields_.end(), [](auto&& field) { return field->valid(); });

  if (nullable()) {
    LEGATE_ASSERT(null_mask()->valid() == result);
  }
  return result;
}

const InternalSharedPtr<PhysicalStore>& StructPhysicalArray::null_mask() const
{
  if (!nullable()) {
    throw TracedException<std::invalid_argument>{
      "Invalid to retrieve the null mask of a non-nullable array"};
  }
  return *null_mask_;  // NOLINT(bugprone-unchecked-optional-access)
}

InternalSharedPtr<PhysicalArray> StructPhysicalArray::child(std::uint32_t index) const
{
  return fields_.at(index);
}

void StructPhysicalArray::populate_stores(
  SmallVector<InternalSharedPtr<PhysicalStore>>& result) const
{
  for (auto&& field : fields_) {
    field->populate_stores(result);
  }
  if (nullable()) {
    result.push_back(null_mask());
  }
}

Domain StructPhysicalArray::domain() const
{
  // Use child() so that the access is bounds-checked in case of empty struct
  return child(0)->domain();
}

void StructPhysicalArray::check_shape_dimension(std::int32_t dim) const
{
  // Use child() so that the access is bounds-checked in case of empty struct
  child(0)->check_shape_dimension(dim);
}

}  // namespace legate::detail
