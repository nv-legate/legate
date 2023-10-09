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

#pragma once

// Useful for IDEs
#include "core/data/accessors.h"

namespace legate {

template <typename VAL>
ListArrayAccessor<LEGION_READ_ONLY, VAL>::ListArrayAccessor(ListArray array)
{
  desc_acc_    = array.descriptor().data().read_accessor<Rect<1>, 1>();
  vardata_acc_ = array.vardata().data().read_accessor<VAL, 1>();
}

template <typename VAL>
ListArrayAccessor<LEGION_READ_ONLY, VAL>::~ListArrayAccessor()
{
}

template <typename VAL>
Span<const VAL> ListArrayAccessor<LEGION_READ_ONLY, VAL>::operator[](const Point<1>& p)
{
  auto& desc = desc_acc_[p];
  return Span<const VAL>(vardata_acc_.ptr(desc.lo), desc.volume());
}

template <typename VAL>
ListArrayAccessor<LEGION_WRITE_DISCARD, VAL>::ListArrayAccessor(ListArray array)
{
  desc_shape_    = array.shape<1>();
  desc_acc_      = array.descriptor().data().write_accessor<Rect<1>, 1>();
  vardata_store_ = array.vardata().data();
}

template <typename VAL>
ListArrayAccessor<LEGION_WRITE_DISCARD, VAL>::~ListArrayAccessor()
{
  if (desc_shape_.empty()) {
    vardata_store_.bind_empty_data();
    return;
  }

  int64_t vardata_size = 0;
  for (auto& value : values_) { vardata_size += value.size(); }
  auto buffer         = vardata_store_.create_output_buffer<VAL>(Point<1>(vardata_size), true);
  auto* p_buffer      = buffer.ptr(0);
  int64_t vardata_pos = 0;
  int64_t desc_pos    = desc_shape_.lo[0];
  for (auto& value : values_) {
    auto len = value.size();
    memcpy(p_buffer, value.data(), len * sizeof(VAL));
    p_buffer += len;

    auto& desc = desc_acc_[desc_pos++];
    desc.lo[0] = vardata_pos;
    desc.hi[0] = vardata_pos + len - 1;
    vardata_pos += len;
  }
}

template <typename VAL>
void ListArrayAccessor<LEGION_WRITE_DISCARD, VAL>::insert(const legate::Span<const VAL>& value)
{
  check_overflow();
  values_.emplace_back(value.begin(), value.end());
}

template <typename VAL>
void ListArrayAccessor<LEGION_WRITE_DISCARD, VAL>::check_overflow()
{
  if (values_.size() >= desc_shape_.volume()) {
    throw std::out_of_range("No space left in the array");
  }
}

StringArrayAccessor<LEGION_READ_ONLY>::StringArrayAccessor(StringArray array)
  : ListArrayAccessor(array.as_list_array())
{
}

std::string_view StringArrayAccessor<LEGION_READ_ONLY>::operator[](const Point<1>& p)
{
  auto span = ListArrayAccessor::operator[](p);
  return std::string_view(reinterpret_cast<const char*>(span.ptr()), span.size());
}

StringArrayAccessor<LEGION_WRITE_DISCARD>::StringArrayAccessor(StringArray array)
  : ListArrayAccessor(array.as_list_array())
{
}

void StringArrayAccessor<LEGION_WRITE_DISCARD>::insert(const std::string& value)
{
  ListArrayAccessor::insert(
    legate::Span<const int8_t>(reinterpret_cast<const int8_t*>(value.data()), value.size()));
}

void StringArrayAccessor<LEGION_WRITE_DISCARD>::insert(const std::string_view& value)
{
  ListArrayAccessor::insert(
    legate::Span<const int8_t>(reinterpret_cast<const int8_t*>(value.data()), value.size()));
}

}  // namespace legate
