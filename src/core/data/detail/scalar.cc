/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "core/data/detail/scalar.h"
#include "core/type/detail/type_info.h"

namespace legate::detail {

Scalar::Scalar(std::shared_ptr<Type> type, const void* data, bool copy)
  : own_(copy), type_(std::move(type)), data_(data)
{
  if (copy) { data_ = copy_data(data, size()); }
}

Scalar::Scalar(const std::string& value) : own_(true), type_(string_type())
{
  auto data_size                  = sizeof(char) * value.size();
  auto buffer                     = malloc(sizeof(uint32_t) + data_size);
  *static_cast<uint32_t*>(buffer) = value.size();
  memcpy(static_cast<int8_t*>(buffer) + sizeof(uint32_t), value.data(), data_size);
  data_ = buffer;
}

Scalar::~Scalar()
{
  if (own_) {
    // We know we own this buffer
    free(const_cast<void*>(data_));
  }
}

Scalar::Scalar(const Scalar& other) : own_(other.own_), type_(other.type_)
{
  if (other.own_) {
    data_ = copy_data(other.data_, other.size());
  } else {
    data_ = other.data_;
  }
}

Scalar::Scalar(Scalar&& other) : own_(other.own_), type_(std::move(other.type_)), data_(other.data_)
{
  other.own_  = false;
  other.data_ = nullptr;
}

Scalar& Scalar::operator=(const Scalar& other)
{
  own_  = other.own_;
  type_ = other.type_;
  if (other.own_) {
    data_ = copy_data(other.data_, other.size());
  } else {
    data_ = other.data_;
  }
  return *this;
}

Scalar& Scalar::operator=(Scalar&& other)
{
  own_        = other.own_;
  type_       = std::move(other.type_);
  data_       = other.data_;
  other.own_  = false;
  other.data_ = nullptr;
  return *this;
}

const void* Scalar::copy_data(const void* data, size_t size)
{
  auto buffer = malloc(size);
  memcpy(buffer, data, size);
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
