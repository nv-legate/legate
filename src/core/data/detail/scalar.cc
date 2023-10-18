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

#include "core/data/detail/scalar.h"
#include "core/type/detail/type_info.h"

#include <type_traits>
#include <utility>

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace legate::detail {

void Scalar::clear_data()
{
  if (own_) {
    // We know we own this buffer
    delete[] const_cast<char*>(static_cast<const char*>(std::exchange(data_, nullptr)));
  }
}

Scalar::Scalar(std::shared_ptr<Type> type, const void* data, bool copy)
  : own_(copy), type_(std::move(type)), data_(data)
{
  if (copy) { data_ = copy_data(data, size()); }
}

Scalar::Scalar(const std::string& value) : own_(true), type_(string_type())
{
  const auto vsize     = static_cast<std::uint32_t>(value.size());
  const auto data_size = sizeof(std::decay_t<decltype(value)>::value_type) * vsize;
  // If you change this, you must also change the pack() function below! The packed buffer must
  // be aligned the same way as it was allocated here, and new char[] aligns to
  // alignof(std::max_align_t)
  const auto buffer = new char[sizeof(vsize) + data_size];

  std::memcpy(buffer, &vsize, sizeof(vsize));
  std::memcpy(buffer + sizeof(vsize), value.data(), data_size);
  data_ = buffer;
}

Scalar::~Scalar() { clear_data(); }

Scalar::Scalar(const Scalar& other) : own_(other.own_), type_(other.type_)
{
  if (other.own_) {
    data_ = copy_data(other.data_, other.size());
  } else {
    data_ = other.data_;
  }
}

Scalar::Scalar(Scalar&& other)
  : own_{std::exchange(other.own_, false)},
    type_{std::move(other.type_)},
    data_{std::exchange(other.data_, nullptr)}
{
}

Scalar& Scalar::operator=(const Scalar& other)
{
  if (this != &other) {
    own_  = other.own_;
    type_ = other.type_;
    clear_data();
    if (other.own_) {
      data_ = copy_data(other.data_, other.size());
    } else {
      data_ = other.data_;
    }
  }
  return *this;
}

Scalar& Scalar::operator=(Scalar&& other)
{
  if (this != &other) {
    own_  = std::exchange(other.own_, false);
    type_ = std::move(other.type_);
    clear_data();
    data_ = std::exchange(other.data_, nullptr);
  }
  return *this;
}

const void* Scalar::copy_data(const void* data, size_t size)
{
  void* buffer = nullptr;

  if (size) {
    buffer = new char[size];
    std::memcpy(buffer, data, size);
  }
  return buffer;
}

size_t Scalar::size() const
{
  if (type_->code == Type::Code::STRING)
    return *static_cast<const uint32_t*>(data_) + sizeof(uint32_t);
  else
    return type_->size();
}

void Scalar::pack(BufferBuilder& buffer) const
{
  type_->pack(buffer);
  buffer.pack_buffer(data_, size());
}

}  // namespace legate::detail
