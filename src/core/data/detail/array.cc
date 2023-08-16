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

namespace legate::detail {

std::shared_ptr<Store> Array::data() const
{
  throw std::invalid_argument("Data store of a nested array cannot be retrieved");
  return nullptr;
}

BaseArray::BaseArray(std::shared_ptr<Store> data, std::shared_ptr<Store> null_mask)
  : data_(std::move(data)), null_mask_(std::move(null_mask))
{
}

bool BaseArray::unbound() const
{
#ifdef DEBUG_LEGATE
  assert(!nullable() || data_->is_unbound_store() == null_mask_->is_unbound_store());
#endif
  return data_->is_unbound_store();
}

bool BaseArray::valid() const { return data_->valid(); }

std::shared_ptr<Store> BaseArray::null_mask() const
{
  if (!nullable()) {
    throw std::invalid_argument("Invalid to retrieve the null mask of a non-nullable array");
  }
  return null_mask_;
}

std::shared_ptr<Array> BaseArray::child(uint32_t index) const
{
  throw std::invalid_argument("Non-nested array has no child sub-array");
  return nullptr;
}

void BaseArray::_stores(std::vector<std::shared_ptr<Store>>& result) const
{
  result.push_back(data_);
  if (nullable()) result.push_back(null_mask_);
}

Domain BaseArray::domain() const { return data_->domain(); }

void BaseArray::check_shape_dimension(const int32_t dim) const
{
  return data_->check_shape_dimension(dim);
}

ListArray::ListArray(std::shared_ptr<Type> type,
                     std::shared_ptr<BaseArray> descriptor,
                     std::shared_ptr<Array> vardata)
  : type_(std::move(type)), descriptor_(std::move(descriptor)), vardata_(std::move(vardata))
{
}

int32_t ListArray::dim() const { return descriptor_->dim(); }

bool ListArray::unbound() const { return descriptor_->unbound() || vardata_->unbound(); }

bool ListArray::valid() const
{
#ifdef DEBUG_LEGATE
  assert(descriptor_->valid() == vardata_->valid());
#endif
  return descriptor_->valid();
}

std::shared_ptr<Array> ListArray::child(uint32_t index) const
{
  switch (index) {
    case 0: return descriptor_;
    case 1: return vardata_;
    default: {
      throw std::out_of_range("List array does not have child " + std::to_string(index));
      break;
    }
  }
  return nullptr;
}

void ListArray::_stores(std::vector<std::shared_ptr<Store>>& result) const
{
  descriptor_->_stores(result);
  vardata_->_stores(result);
}

Domain ListArray::domain() const { return descriptor_->domain(); }

void ListArray::check_shape_dimension(const int32_t dim) const
{
  descriptor_->check_shape_dimension(dim);
}

StructArray::StructArray(std::shared_ptr<Type> type,
                         std::shared_ptr<Store> null_mask,
                         std::vector<std::shared_ptr<Array>>&& fields)
  : type_(std::move(type)), null_mask_(std::move(null_mask)), fields_(std::move(fields))
{
}

int32_t StructArray::dim() const { return fields_.front()->dim(); }

bool StructArray::unbound() const
{
  return std::any_of(fields_.begin(), fields_.end(), [](auto& field) { return field->unbound(); });
}

bool StructArray::valid() const
{
  auto result =
    std::all_of(fields_.begin(), fields_.end(), [](auto& field) { return field->valid(); });
#ifdef DEBUG_LEGATE
  assert(null_mask_->valid() == result);
#endif
  return result;
}

std::shared_ptr<Store> StructArray::null_mask() const
{
  if (!nullable()) {
    throw std::invalid_argument("Invalid to retrieve the null mask of a non-nullable array");
  }
  return null_mask_;
}

std::shared_ptr<Array> StructArray::child(uint32_t index) const { return fields_.at(index); }

void StructArray::_stores(std::vector<std::shared_ptr<Store>>& result) const
{
  std::for_each(fields_.begin(), fields_.end(), [&result](auto& field) { field->_stores(result); });
}

Domain StructArray::domain() const { return fields_.front()->domain(); }

void StructArray::check_shape_dimension(const int32_t dim) const
{
  fields_.front()->check_shape_dimension(dim);
}

}  // namespace legate::detail
