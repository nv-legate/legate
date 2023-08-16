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

#include "core/data/array.h"
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
class ListArrayAccessor<READ_ONLY, VAL> {
 public:
  ListArrayAccessor(ListArray array);
  virtual ~ListArrayAccessor();

 public:
  Span<const VAL> operator[](const Point<1>& p);

 private:
  AccessorRO<Rect<1>, 1> desc_acc_;
  AccessorRO<VAL, 1> vardata_acc_;
};

template <typename VAL>
class ListArrayAccessor<WRITE_DISCARD, VAL> {
 public:
  ListArrayAccessor(ListArray array);
  virtual ~ListArrayAccessor();

 public:
  void insert(const legate::Span<const VAL>& value);

 private:
  void check_overflow();

 private:
  Rect<1> desc_shape_;
  AccessorWO<Rect<1>, 1> desc_acc_;
  Store vardata_store_;
  std::vector<std::vector<VAL>> values_{};
};

template <Legion::PrivilegeMode>
class StringArrayAccessor;

template <>
struct StringArrayAccessor<READ_ONLY> : public ListArrayAccessor<READ_ONLY, int8_t> {
  StringArrayAccessor(StringArray array);

  using ListArrayAccessor::operator[];
  std::string_view operator[](const Point<1>& p);
};

template <>
struct StringArrayAccessor<WRITE_DISCARD> : public ListArrayAccessor<WRITE_DISCARD, int8_t> {
  StringArrayAccessor(StringArray array);

  using ListArrayAccessor::insert;
  void insert(const std::string& value);
  void insert(const std::string_view& value);
};

}  // namespace legate

#include "core/data/accessors.inl"
