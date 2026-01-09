/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/array.h>

#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>

namespace legate::mapping::detail {

const InternalSharedPtr<Store>& Array::data() const
{
  throw legate::detail::TracedException<std::invalid_argument>{
    "Data store of a nested array cannot be retrieved"};

  static const InternalSharedPtr<Store> ptr;
  return ptr;
}

// ==========================================================================================

bool BaseArray::unbound() const
{
  LEGATE_ASSERT(!nullable() || data()->unbound() == null_mask()->unbound());
  return data()->unbound();
}

const InternalSharedPtr<Store>& BaseArray::null_mask() const
{
  if (!nullable()) {
    throw legate::detail::TracedException<std::invalid_argument>{
      "Invalid to retrieve the null mask of a non-nullable array"};
  }
  return *null_mask_;  // NOLINT(bugprone-unchecked-optional-access)
}

InternalSharedPtr<Array> BaseArray::child(std::uint32_t /*index*/) const
{
  throw legate::detail::TracedException<std::invalid_argument>{
    "Non-nested array has no child sub-array"};
  return {};
}

void BaseArray::populate_stores(legate::detail::SmallVector<InternalSharedPtr<Store>>& result) const
{
  result.push_back(data());
  if (nullable()) {
    result.push_back(null_mask());
  }
}

Domain BaseArray::domain() const { return data()->domain(); }

// ==========================================================================================

std::int32_t ListArray::dim() const { return descriptor()->dim(); }

bool ListArray::unbound() const { return descriptor()->unbound() || vardata()->unbound(); }

bool ListArray::valid() const
{
  LEGATE_ASSERT(descriptor()->valid() == vardata()->valid());
  return descriptor()->valid();
}

InternalSharedPtr<Array> ListArray::child(std::uint32_t index) const
{
  switch (index) {
    case 0: return descriptor();
    case 1: return vardata();
    default: {  // legate-lint: no-switch-default
      throw legate::detail::TracedException<std::out_of_range>{
        fmt::format("List array does not have child {}", index)};
      break;
    }
  }
  return {};
}

void ListArray::populate_stores(legate::detail::SmallVector<InternalSharedPtr<Store>>& result) const
{
  descriptor()->populate_stores(result);
  vardata()->populate_stores(result);
}

Domain ListArray::domain() const { return descriptor()->domain(); }

// ==========================================================================================

std::int32_t StructArray::dim() const { return fields().front()->dim(); }

bool StructArray::unbound() const
{
  return std::any_of(
    fields().begin(), fields().end(), [](const auto& field) { return field->unbound(); });
}

bool StructArray::valid() const
{
  const auto result =
    std::all_of(fields().begin(), fields().end(), [](const auto& field) { return field->valid(); });

  if (nullable()) {
    LEGATE_ASSERT(null_mask()->valid() == result);
  }
  return result;
}

const InternalSharedPtr<Store>& StructArray::null_mask() const
{
  if (!nullable()) {
    throw legate::detail::TracedException<std::invalid_argument>{
      "Invalid to retrieve the null mask of a non-nullable array"};
  }
  return *null_mask_;  // NOLINT(bugprone-unchecked-optional-access)
}

InternalSharedPtr<Array> StructArray::child(std::uint32_t index) const
{
  return fields().at(index);
}

void StructArray::populate_stores(
  legate::detail::SmallVector<InternalSharedPtr<Store>>& result) const
{
  for (auto&& field : fields()) {
    field->populate_stores(result);
  }
}

Domain StructArray::domain() const
{
  // Use child() so access is bounds-checked
  return child(0)->domain();
}

}  // namespace legate::mapping::detail
