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

#include "core/data/physical_array.h"
#include "core/utilities/span.h"
#include "core/utilities/typedefs.h"

/**
 * @file
 * @brief Class definitions for array accessors
 */

namespace legate {

template <Legion::PrivilegeMode, typename VAL>
class ListArrayAccessor;

template <typename VAL>
class ListArrayAccessor<LEGION_READ_ONLY, VAL> {
 public:
  ListArrayAccessor(const ListArray& array);
  virtual ~ListArrayAccessor() = default;

  Span<const VAL> operator[](const Point<1>& p);

 private:
  AccessorRO<Rect<1>, 1> desc_acc_;
  AccessorRO<VAL, 1> vardata_acc_;
};

template <typename VAL>
class ListArrayAccessor<LEGION_WRITE_DISCARD, VAL> {
 public:
  ListArrayAccessor(const ListArray& array);
  virtual ~ListArrayAccessor() noexcept;

  void insert(const legate::Span<const VAL>& value);

 private:
  void check_overflow();

  Rect<1> desc_shape_;
  AccessorWO<Rect<1>, 1> desc_acc_;
  Store vardata_store_;
  std::vector<std::vector<VAL>> values_{};
};

template <Legion::PrivilegeMode>
class StringArrayAccessor;

template <>
class StringArrayAccessor<LEGION_READ_ONLY>
  : public ListArrayAccessor<LEGION_READ_ONLY, std::int8_t> {
  StringArrayAccessor(const StringArray& array);

  using ListArrayAccessor::operator[];
  std::string_view operator[](const Point<1>& p);
};

template <>
class StringArrayAccessor<LEGION_WRITE_DISCARD>
  : public ListArrayAccessor<LEGION_WRITE_DISCARD, std::int8_t> {
  StringArrayAccessor(const StringArray& array);

  using ListArrayAccessor::insert;
  void insert(std::string_view value);
};

}  // namespace legate

#include "core/data/accessors.inl"
