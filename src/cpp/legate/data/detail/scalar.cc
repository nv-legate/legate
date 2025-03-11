/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/scalar.h>

#include <legate/type/detail/types.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

namespace legate::detail {

void Scalar::clear_data_()
{
  if (own_) {
    // We know we own this buffer
    delete[] const_cast<char*>(static_cast<const char*>(std::exchange(data_, nullptr)));
  }
}

Scalar::~Scalar() { clear_data_(); }

Scalar::Scalar(InternalSharedPtr<Type> type, const void* data, bool copy)
  : own_{data && copy}, type_{std::move(type)}, data_{data}
{
  if (own_) {
    // Data of course can never be NULL at this point, given that own_ = data && copy, but
    // clang-tidy (or specifically the clang static analyzer) appears to think that it could
    // be. My best guess is that it believes the move ctor type could somehow gain access to
    // the this pointer for the Scalar being constructed (possibly through the void *???).
    //
    // In any case, this LEGATE_ASSERT() exists only to silence clang-tidy.
    LEGATE_ASSERT(data != nullptr);
    data_ = copy_data_(data, size());
  }
}

Scalar::Scalar(std::string_view value) : own_{true}, type_{string_type()}
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

Scalar& Scalar::operator=(const Scalar& other)
{
  if (this != &other) {
    own_  = other.own_;
    type_ = other.type_;
    clear_data_();
    if (other.own_) {
      data_ = copy_data_(other.data_, other.size());
    } else {
      data_ = other.data_;
    }
  }
  return *this;
}

Scalar& Scalar::operator=(Scalar&& other) noexcept
{
  if (this != &other) {
    own_  = std::exchange(other.own_, false);
    type_ = std::move(other.type_);
    clear_data_();
    data_ = std::exchange(other.data_, nullptr);
  }
  return *this;
}

const void* Scalar::copy_data_(const void* data, std::size_t size)
{
  void* buffer = nullptr;

  if (size) {
    buffer = new char[size];
    std::memcpy(buffer, data, size);
  }
  return buffer;
}

std::size_t Scalar::size() const
{
  if (type()->code == Type::Code::STRING) {
    return *static_cast<const std::uint32_t*>(data()) + sizeof(std::uint32_t);
  }
  return type()->size();
}

void Scalar::pack(BufferBuilder& buffer) const
{
  type()->pack(buffer);
  buffer.pack_buffer(data(), size(), type()->alignment());
}

}  // namespace legate::detail
