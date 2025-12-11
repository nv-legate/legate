/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/physical_arrays/base_physical_array.h>

#include <legate/data/detail/physical_stores/unbound_physical_store.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

namespace legate::detail {

bool BasePhysicalArray::unbound() const
{
  const auto data_unbound = dynamic_cast<const UnboundPhysicalStore*>(data().get()) != nullptr;

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG) && nullable()) {
    const auto null_mask_unbound =
      dynamic_cast<const UnboundPhysicalStore*>(null_mask().get()) != nullptr;

    LEGATE_CHECK(data_unbound == null_mask_unbound);
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
  data()->check_shape_dimension(dim);
}

}  // namespace legate::detail
