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

#include <memory>

#include "core/type/detail/type_info.h"
#include "core/type/type_traits.h"
#include "core/utilities/detail/buffer_builder.h"

namespace legate::detail {

class Type;

class Scalar {
 public:
  Scalar(std::shared_ptr<Type> type, const void* data, bool copy);
  Scalar(const std::string& value);
  ~Scalar();

 public:
  template <typename T>
  Scalar(T value) : own_(true), type_(detail::primitive_type(legate_type_code_of<T>))
  {
    static_assert(legate_type_code_of<T> != Type::Code::FIXED_ARRAY);
    static_assert(legate_type_code_of<T> != Type::Code::STRUCT);
    static_assert(legate_type_code_of<T> != Type::Code::STRING);
    static_assert(legate_type_code_of<T> != Type::Code::INVALID);
    data_ = copy_data(&value, sizeof(T));
  }

 public:
  Scalar(const Scalar& other);
  Scalar(Scalar&& other);

 public:
  Scalar& operator=(const Scalar& other);
  Scalar& operator=(Scalar&& other);

 private:
  const void* copy_data(const void* data, size_t size);

 public:
  std::shared_ptr<Type> type() const { return type_; }
  const void* data() const { return data_; }
  size_t size() const;

 public:
  void pack(BufferBuilder& buffer) const;

 private:
  bool own_;
  std::shared_ptr<Type> type_;
  const void* data_;
};

}  // namespace legate::detail
