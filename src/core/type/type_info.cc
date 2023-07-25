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

#include "core/type/type_info.h"
#include "core/runtime/detail/runtime.h"
#include "core/type/detail/type_info.h"

namespace legate {

Type::Code Type::code() const { return impl_->code; }

uint32_t Type::size() const { return impl_->size(); }

uint32_t Type::alignment() const { return impl_->alignment(); }

int32_t Type::uid() const { return impl_->uid(); }

bool Type::variable_size() const { return impl_->variable_size(); }

std::string Type::to_string() const { return impl_->to_string(); }

bool Type::is_primitive() const { return impl_->is_primitive(); }

FixedArrayType Type::as_fixed_array_type() const
{
  if (impl_->code != Code::FIXED_ARRAY) {
    throw std::invalid_argument("Type is not a fixed array type");
  }
  return FixedArrayType(impl_);
}

StructType Type::as_struct_type() const
{
  if (impl_->code != Code::STRUCT) { throw std::invalid_argument("Type is not a struct type"); }
  return StructType(impl_);
}

void Type::record_reduction_operator(int32_t op_kind, int32_t global_op_id) const
{
  impl_->record_reduction_operator(op_kind, global_op_id);
}

int32_t Type::find_reduction_operator(int32_t op_kind) const
{
  return impl_->find_reduction_operator(op_kind);
}

int32_t Type::find_reduction_operator(ReductionOpKind op_kind) const
{
  return impl_->find_reduction_operator(op_kind);
}

bool Type::operator==(const Type& other) const { return impl_->equal(*other.impl_); }

bool Type::operator!=(const Type& other) const { return !operator==(other); }

Type::Type() {}

Type::Type(std::shared_ptr<detail::Type> impl) : impl_(std::move(impl)) {}

Type::~Type() {}

uint32_t FixedArrayType::num_elements() const
{
  return static_cast<const detail::FixedArrayType*>(impl_.get())->num_elements();
}

Type FixedArrayType::element_type() const
{
  return Type(static_cast<const detail::FixedArrayType*>(impl_.get())->element_type());
}

FixedArrayType::FixedArrayType(std::shared_ptr<detail::Type> type) : Type(std::move(type)) {}

uint32_t StructType::num_fields() const
{
  return static_cast<const detail::StructType*>(impl_.get())->num_fields();
}

Type StructType::field_type(uint32_t field_idx) const
{
  return Type(static_cast<const detail::StructType*>(impl_.get())->field_type(field_idx));
}

bool StructType::aligned() const
{
  return static_cast<const detail::StructType*>(impl_.get())->aligned();
}

StructType::StructType(std::shared_ptr<detail::Type> type) : Type(std::move(type)) {}

Type primitive_type(Type::Code code) { return Type(detail::primitive_type(code)); }

Type string_type() { return Type(detail::string_type()); }

Type fixed_array_type(const Type& element_type, uint32_t N) noexcept(false)
{
  return Type(detail::fixed_array_type(element_type.impl(), N));
}

Type struct_type(const std::vector<Type>& field_types, bool align) noexcept(false)
{
  std::vector<std::shared_ptr<detail::Type>> detail_field_types;
  for (const auto& field_type : field_types) { detail_field_types.push_back(field_type.impl()); }
  return Type(detail::struct_type(std::move(detail_field_types), align));
}

std::ostream& operator<<(std::ostream& ostream, const Type::Code& code)
{
  ostream << static_cast<int32_t>(code);
  return ostream;
}

std::ostream& operator<<(std::ostream& ostream, const Type& type)
{
  ostream << type.to_string();
  return ostream;
}

Type bool_()
{
  static auto result = primitive_type(Type::Code::BOOL);
  return result;
}

Type int8()
{
  static auto result = primitive_type(Type::Code::INT8);
  return result;
}

Type int16()
{
  static auto result = primitive_type(Type::Code::INT16);
  return result;
}

Type int32()
{
  static auto result = primitive_type(Type::Code::INT32);
  return result;
}

Type int64()
{
  static auto result = primitive_type(Type::Code::INT64);
  return result;
}

Type uint8()
{
  static auto result = primitive_type(Type::Code::UINT8);
  return result;
}

Type uint16()
{
  static auto result = primitive_type(Type::Code::UINT16);
  return result;
}

Type uint32()
{
  static auto result = primitive_type(Type::Code::UINT32);
  return result;
}

Type uint64()
{
  static auto result = primitive_type(Type::Code::UINT64);
  return result;
}

Type float16()
{
  static auto result = primitive_type(Type::Code::FLOAT16);
  return result;
}

Type float32()
{
  static auto result = primitive_type(Type::Code::FLOAT32);
  return result;
}

Type float64()
{
  static auto result = primitive_type(Type::Code::FLOAT64);
  return result;
}

Type complex64()
{
  static auto result = primitive_type(Type::Code::COMPLEX64);
  return result;
}

Type complex128()
{
  static auto result = primitive_type(Type::Code::COMPLEX128);
  return result;
}

namespace {

constexpr int32_t POINT_UID_BASE = static_cast<int32_t>(Type::Code::INVALID);
constexpr int32_t RECT_UID_BASE  = POINT_UID_BASE + LEGATE_MAX_DIM + 1;

}  // namespace

Type point_type(int32_t ndim)
{
  static Type cache[LEGATE_MAX_DIM + 1];

  if (ndim <= 0 || ndim > LEGATE_MAX_DIM)
    throw std::out_of_range(std::to_string(ndim) + " is not a supported number of dimensions");
  if (cache[ndim].impl() == nullptr) {
    cache[ndim] =
      Type(std::make_shared<detail::FixedArrayType>(POINT_UID_BASE + ndim, int64().impl(), ndim));
  }
  return cache[ndim];
}

Type rect_type(int32_t ndim)
{
  std::vector<std::shared_ptr<detail::Type>> field_types{point_type(ndim).impl(),
                                                         point_type(ndim).impl()};
  return Type(std::make_shared<detail::StructType>(
    RECT_UID_BASE + ndim, std::move(field_types), true /*align*/));
}

bool is_point_type(const Type& type, int32_t ndim)
{
  switch (type.code()) {
    case Type::Code::INT64: {
      return 1 == ndim;
    }
    case Type::Code::FIXED_ARRAY: {
      return type.as_fixed_array_type().num_elements() == ndim;
    }
    default: {
      return false;
    }
  }
  return false;
}

bool is_rect_type(const Type& type, int32_t ndim)
{
  if (type.code() != Type::Code::STRUCT) return false;
  auto st_type = type.as_struct_type();
  return st_type.num_fields() == 2 && is_point_type(st_type.field_type(0), ndim) &&
         is_point_type(st_type.field_type(1), ndim);
}

}  // namespace legate
