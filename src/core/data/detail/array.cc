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

#include "core/data/detail/array.h"

#include <stdexcept>

namespace legate::detail {

std::shared_ptr<Store> Array::data() const
{
  throw std::invalid_argument{"Data store of a nested array cannot be retrieved"};
  return {};
}

bool BaseArray::unbound() const
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(!nullable() || data_->is_unbound_store() == null_mask_->is_unbound_store());
  }
  return data_->is_unbound_store();
}

std::shared_ptr<Store> BaseArray::null_mask() const
{
  if (!nullable()) {
    throw std::invalid_argument{"Invalid to retrieve the null mask of a non-nullable array"};
  }
  return null_mask_;
}

std::shared_ptr<Array> BaseArray::child(uint32_t /*index*/) const
{
  throw std::invalid_argument{"Non-nested array has no child sub-array"};
  return {};
}

void BaseArray::_stores(std::vector<std::shared_ptr<Store>>& result) const
{
  result.push_back(data_);
  if (nullable()) result.push_back(null_mask_);
}

void BaseArray::check_shape_dimension(int32_t dim) const
{
  return data_->check_shape_dimension(dim);
}

bool ListArray::valid() const
{
  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(descriptor_->valid() == vardata_->valid()); }
  return descriptor_->valid();
}

std::shared_ptr<Array> ListArray::child(uint32_t index) const
{
  switch (index) {
    case 0: return descriptor_;
    case 1: return vardata_;
    default: {
      throw std::out_of_range{"List array does not have child " + std::to_string(index)};
      break;
    }
  }
  return {};
}

void ListArray::_stores(std::vector<std::shared_ptr<Store>>& result) const
{
  descriptor_->_stores(result);
  vardata_->_stores(result);
}

void ListArray::check_shape_dimension(int32_t dim) const
{
  descriptor_->check_shape_dimension(dim);
}

bool StructArray::unbound() const
{
  return std::any_of(fields_.begin(), fields_.end(), [](auto& field) { return field->unbound(); });
}

bool StructArray::valid() const
{
  auto result =
    std::all_of(fields_.begin(), fields_.end(), [](auto& field) { return field->valid(); });
  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(null_mask_->valid() == result); }
  return result;
}

std::shared_ptr<Store> StructArray::null_mask() const
{
  if (!nullable()) {
    throw std::invalid_argument{"Invalid to retrieve the null mask of a non-nullable array"};
  }
  return null_mask_;
}

std::shared_ptr<Array> StructArray::child(uint32_t index) const { return fields_.at(index); }

void StructArray::_stores(std::vector<std::shared_ptr<Store>>& result) const
{
  std::for_each(fields_.begin(), fields_.end(), [&result](auto& field) { field->_stores(result); });
}

Domain StructArray::domain() const { return fields_.front()->domain(); }

void StructArray::check_shape_dimension(int32_t dim) const
{
  fields_.front()->check_shape_dimension(dim);
}

}  // namespace legate::detail
