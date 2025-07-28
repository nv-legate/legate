/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/physical_array.h>

#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>

namespace legate::detail {

const InternalSharedPtr<PhysicalStore>& PhysicalArray::data() const
{
  throw TracedException<std::invalid_argument>{"Data store of a nested array cannot be retrieved"};
}

// ==========================================================================================

bool BasePhysicalArray::unbound() const
{
  const auto data_unbound = data()->is_unbound_store();

  if (nullable()) {
    LEGATE_ASSERT(data_unbound == null_mask()->is_unbound_store());
  }
  return data_unbound;
}

const InternalSharedPtr<PhysicalStore>& BasePhysicalArray::null_mask() const
{
  if (!nullable()) {
    throw TracedException<std::invalid_argument>{
      "Invalid to retrieve the null mask of a non-nullable array"};
  }
  return *null_mask_;  // NOLINT(bugprone-unchecked-optional-access)
}

InternalSharedPtr<PhysicalArray> BasePhysicalArray::child(std::uint32_t /*index*/) const
{
  throw TracedException<std::invalid_argument>{"Non-nested array has no child sub-array"};
}

void BasePhysicalArray::populate_stores(SmallVector<InternalSharedPtr<PhysicalStore>>& result) const
{
  result.push_back(data());
  if (nullable()) {
    result.push_back(null_mask());
  }
}

void BasePhysicalArray::check_shape_dimension(std::int32_t dim) const
{
  data()->check_shape_dimension_(dim);
}

// ==========================================================================================

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

// ==========================================================================================

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
