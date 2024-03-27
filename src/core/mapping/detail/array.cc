/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/mapping/detail/array.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace legate::mapping::detail {

InternalSharedPtr<Store> Array::data() const
{
  throw std::invalid_argument("Data store of a nested array cannot be retrieved");
  return {};
}

bool BaseArray::unbound() const
{
  LegateAssert(!nullable() || data_->unbound() == null_mask_->unbound());
  return data_->unbound();
}

InternalSharedPtr<Store> BaseArray::null_mask() const
{
  if (!nullable()) {
    throw std::invalid_argument("Invalid to retrieve the null mask of a non-nullable array");
  }
  return null_mask_;
}

InternalSharedPtr<Array> BaseArray::child(std::uint32_t /*index*/) const
{
  throw std::invalid_argument("Non-nested array has no child sub-array");
  return {};
}

void BaseArray::_stores(std::vector<InternalSharedPtr<Store>>& result) const
{
  result.push_back(data_);
  if (nullable()) {
    result.push_back(null_mask_);
  }
}

Domain BaseArray::domain() const { return data_->domain(); }

std::int32_t ListArray::dim() const { return descriptor_->dim(); }

bool ListArray::unbound() const { return descriptor_->unbound() || vardata_->unbound(); }

InternalSharedPtr<Array> ListArray::child(std::uint32_t index) const
{
  switch (index) {
    case 0: return descriptor_;
    case 1: return vardata_;
    default: {
      throw std::out_of_range("List array does not have child " + std::to_string(index));
      break;
    }
  }
  return {};
}

void ListArray::_stores(std::vector<InternalSharedPtr<Store>>& result) const
{
  descriptor_->_stores(result);
  vardata_->_stores(result);
}

Domain ListArray::domain() const { return descriptor_->domain(); }

std::int32_t StructArray::dim() const { return fields_.front()->dim(); }

bool StructArray::unbound() const
{
  return std::any_of(fields_.begin(), fields_.end(), [](auto& field) { return field->unbound(); });
}

InternalSharedPtr<Store> StructArray::null_mask() const
{
  if (!nullable()) {
    throw std::invalid_argument("Invalid to retrieve the null mask of a non-nullable array");
  }
  return null_mask_;
}

InternalSharedPtr<Array> StructArray::child(std::uint32_t index) const { return fields_.at(index); }

void StructArray::_stores(std::vector<InternalSharedPtr<Store>>& result) const
{
  std::for_each(fields_.begin(), fields_.end(), [&result](auto& field) { field->_stores(result); });
}

Domain StructArray::domain() const { return fields_.front()->domain(); }

}  // namespace legate::mapping::detail
