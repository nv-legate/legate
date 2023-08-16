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

ListType Type::as_list_type() const
{
  if (impl_->code != Code::LIST) { throw std::invalid_argument("Type is not a list type"); }
  return ListType(impl_);
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

Type ListType::element_type() const
{
  return Type(static_cast<const detail::ListType*>(impl_.get())->element_type());
}

ListType::ListType(std::shared_ptr<detail::Type> type) : Type(std::move(type)) {}

Type primitive_type(Type::Code code) { return Type(detail::primitive_type(code)); }

Type string_type() { return Type(detail::string_type()); }

Type fixed_array_type(const Type& element_type, uint32_t N)
{
  return Type(detail::fixed_array_type(element_type.impl(), N));
}

Type struct_type(const std::vector<Type>& field_types, bool align)
{
  std::vector<std::shared_ptr<detail::Type>> detail_field_types;
  for (const auto& field_type : field_types) { detail_field_types.push_back(field_type.impl()); }
  return Type(detail::struct_type(std::move(detail_field_types), align));
}

Type list_type(const Type& element_type) { return Type(detail::list_type(element_type.impl())); }

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

Type bool_() { return Type(detail::bool_()); }

Type int8() { return Type(detail::int8()); }

Type int16() { return Type(detail::int16()); }

Type int32() { return Type(detail::int32()); }

Type int64() { return Type(detail::int64()); }

Type uint8() { return Type(detail::uint8()); }

Type uint16() { return Type(detail::uint16()); }

Type uint32() { return Type(detail::uint32()); }

Type uint64() { return Type(detail::uint64()); }

Type float16() { return Type(detail::float16()); }

Type float32() { return Type(detail::float32()); }

Type float64() { return Type(detail::float64()); }

Type complex64() { return Type(detail::complex64()); }

Type complex128() { return Type(detail::complex128()); }

Type point_type(int32_t ndim) { return Type(detail::point_type(ndim)); }

Type rect_type(int32_t ndim) { return Type(detail::rect_type(ndim)); }

bool is_point_type(const Type& type, int32_t ndim)
{
  return detail::is_point_type(type.impl(), ndim);
}

bool is_rect_type(const Type& type, int32_t ndim)
{
  return detail::is_rect_type(type.impl(), ndim);
}

}  // namespace legate
