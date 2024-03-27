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

#include "core/type/type_info.h"

#include "core/type/detail/type_info.h"

#include <iostream>
#include <stdexcept>

namespace legate {

Type::Code Type::code() const { return impl()->code; }

std::uint32_t Type::size() const { return impl()->size(); }

std::uint32_t Type::alignment() const { return impl()->alignment(); }

std::uint32_t Type::uid() const { return impl()->uid(); }

bool Type::variable_size() const { return impl()->variable_size(); }

std::string Type::to_string() const { return impl()->to_string(); }

bool Type::is_primitive() const { return impl()->is_primitive(); }

FixedArrayType Type::as_fixed_array_type() const
{
  if (code() != Code::FIXED_ARRAY) {
    throw std::invalid_argument{"Type is not a fixed array type"};
  }
  return FixedArrayType{impl()};
}

StructType Type::as_struct_type() const
{
  if (code() != Code::STRUCT) {
    throw std::invalid_argument{"Type is not a struct type"};
  }
  return StructType{impl()};
}

ListType Type::as_list_type() const
{
  if (code() != Code::LIST) {
    throw std::invalid_argument{"Type is not a list type"};
  }
  return ListType{impl()};
}

void Type::record_reduction_operator(std::int32_t op_kind, std::int32_t global_op_id) const
{
  impl()->record_reduction_operator(op_kind, global_op_id);
}

void Type::record_reduction_operator(ReductionOpKind op_kind, std::int32_t global_op_id) const
{
  impl()->record_reduction_operator(static_cast<std::int32_t>(op_kind), global_op_id);
}

std::int32_t Type::find_reduction_operator(std::int32_t op_kind) const
{
  return impl()->find_reduction_operator(op_kind);
}

std::int32_t Type::find_reduction_operator(ReductionOpKind op_kind) const
{
  return impl()->find_reduction_operator(static_cast<std::int32_t>(op_kind));
}

bool Type::operator==(const Type& other) const { return impl()->equal(*other.impl_); }

bool Type::operator!=(const Type& other) const { return !operator==(other); }

Type::Type() = default;

Type::~Type() = default;

std::uint32_t FixedArrayType::num_elements() const
{
  return static_cast<const detail::FixedArrayType*>(impl().get())->num_elements();
}

Type FixedArrayType::element_type() const
{
  return Type{static_cast<const detail::FixedArrayType*>(impl().get())->element_type()};
}

std::uint32_t StructType::num_fields() const
{
  return static_cast<const detail::StructType*>(impl().get())->num_fields();
}

Type StructType::field_type(std::uint32_t field_idx) const
{
  return Type{static_cast<const detail::StructType*>(impl().get())->field_type(field_idx)};
}

bool StructType::aligned() const
{
  return static_cast<const detail::StructType*>(impl().get())->aligned();
}

std::vector<std::uint32_t> StructType::offsets() const
{
  return static_cast<const detail::StructType*>(impl().get())->offsets();
}

Type ListType::element_type() const
{
  return Type{static_cast<const detail::ListType*>(impl().get())->element_type()};
}

Type primitive_type(Type::Code code) { return Type{detail::primitive_type(code)}; }

Type string_type() { return Type{detail::string_type()}; }

Type fixed_array_type(const Type& element_type, std::uint32_t N)
{
  return Type{detail::fixed_array_type(element_type.impl(), N)};
}

Type struct_type(const std::vector<Type>& field_types, bool align)
{
  std::vector<InternalSharedPtr<detail::Type>> detail_field_types;

  detail_field_types.reserve(field_types.size());
  for (const auto& field_type : field_types) {
    detail_field_types.emplace_back(field_type.impl());
  }
  return Type{detail::struct_type(std::move(detail_field_types), align)};
}

Type list_type(const Type& element_type) { return Type{detail::list_type(element_type.impl())}; }

std::ostream& operator<<(std::ostream& ostream, const Type::Code& code)
{
  ostream << static_cast<std::int32_t>(code);
  return ostream;
}

std::ostream& operator<<(std::ostream& ostream, const Type& type)
{
  ostream << type.to_string();
  return ostream;
}

Type bool_() { return Type{detail::bool_()}; }

Type int8() { return Type{detail::int8()}; }

Type int16() { return Type{detail::int16()}; }

Type int32() { return Type{detail::int32()}; }

Type int64() { return Type{detail::int64()}; }

Type uint8() { return Type{detail::uint8()}; }

Type uint16() { return Type{detail::uint16()}; }

Type uint32() { return Type{detail::uint32()}; }

Type uint64() { return Type{detail::uint64()}; }

Type float16() { return Type{detail::float16()}; }

Type float32() { return Type{detail::float32()}; }

Type float64() { return Type{detail::float64()}; }

Type complex64() { return Type{detail::complex64()}; }

Type complex128() { return Type{detail::complex128()}; }

Type binary_type(std::uint32_t size) { return Type{detail::binary_type(size)}; }

Type point_type(std::uint32_t ndim) { return Type{detail::point_type(ndim)}; }

Type rect_type(std::uint32_t ndim) { return Type{detail::rect_type(ndim)}; }

Type null_type() { return Type{detail::null_type()}; }

bool is_point_type(const Type& type, std::uint32_t ndim)
{
  return detail::is_point_type(type.impl(), ndim);
}

bool is_rect_type(const Type& type, std::uint32_t ndim)
{
  return detail::is_rect_type(type.impl(), ndim);
}

}  // namespace legate
