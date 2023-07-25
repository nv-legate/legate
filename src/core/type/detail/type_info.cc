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

#include <numeric>
#include <unordered_map>

#include "core/runtime/detail/runtime.h"
#include "core/type/detail/type_info.h"
#include "core/utilities/detail/buffer_builder.h"

namespace legate::detail {

namespace {

const std::unordered_map<Type::Code, uint32_t> SIZEOF = {
  {Type::Code::BOOL, sizeof(legate_type_of<Type::Code::BOOL>)},
  {Type::Code::INT8, sizeof(legate_type_of<Type::Code::INT8>)},
  {Type::Code::INT16, sizeof(legate_type_of<Type::Code::INT16>)},
  {Type::Code::INT32, sizeof(legate_type_of<Type::Code::INT32>)},
  {Type::Code::INT64, sizeof(legate_type_of<Type::Code::INT64>)},
  {Type::Code::UINT8, sizeof(legate_type_of<Type::Code::UINT8>)},
  {Type::Code::UINT16, sizeof(legate_type_of<Type::Code::UINT16>)},
  {Type::Code::UINT32, sizeof(legate_type_of<Type::Code::UINT32>)},
  {Type::Code::UINT64, sizeof(legate_type_of<Type::Code::UINT64>)},
  {Type::Code::FLOAT16, sizeof(legate_type_of<Type::Code::FLOAT16>)},
  {Type::Code::FLOAT32, sizeof(legate_type_of<Type::Code::FLOAT32>)},
  {Type::Code::FLOAT64, sizeof(legate_type_of<Type::Code::FLOAT64>)},
  {Type::Code::COMPLEX64, sizeof(legate_type_of<Type::Code::COMPLEX64>)},
  {Type::Code::COMPLEX128, sizeof(legate_type_of<Type::Code::COMPLEX128>)},
};

const std::unordered_map<Type::Code, std::string> TYPE_NAMES = {
  {Type::Code::BOOL, "bool"},
  {Type::Code::INT8, "int8"},
  {Type::Code::INT16, "int16"},
  {Type::Code::INT32, "int32"},
  {Type::Code::INT64, "int64"},
  {Type::Code::UINT8, "uint8"},
  {Type::Code::UINT16, "uint16"},
  {Type::Code::UINT32, "uint32"},
  {Type::Code::UINT64, "uint64"},
  {Type::Code::FLOAT16, "float16"},
  {Type::Code::FLOAT32, "float32"},
  {Type::Code::FLOAT64, "float64"},
  {Type::Code::COMPLEX64, "complex64"},
  {Type::Code::COMPLEX128, "complex128"},
  {Type::Code::STRING, "string"},
};

const char* _VARIABLE_SIZE_ERROR_MESSAGE = "Variable-size element type cannot be used";

}  // namespace

Type::Type(Code c) : code(c) {}

const FixedArrayType& Type::as_fixed_array_type() const
{
  throw std::invalid_argument("Type is not a fixed array type");
  return *static_cast<const FixedArrayType*>(nullptr);
}

const StructType& Type::as_struct_type() const
{
  throw std::invalid_argument("Type is not a struct type");
  return *static_cast<const StructType*>(nullptr);
}

void Type::record_reduction_operator(int32_t op_kind, int32_t global_op_id) const
{
  detail::Runtime::get_runtime()->record_reduction_operator(uid(), op_kind, global_op_id);
}

int32_t Type::find_reduction_operator(int32_t op_kind) const
{
  return detail::Runtime::get_runtime()->find_reduction_operator(uid(), op_kind);
}

int32_t Type::find_reduction_operator(ReductionOpKind op_kind) const
{
  return find_reduction_operator(static_cast<int32_t>(op_kind));
}

bool Type::operator==(const Type& other) const { return equal(other); }

PrimitiveType::PrimitiveType(Code code) : Type(code), size_(SIZEOF.at(code)) {}

int32_t PrimitiveType::uid() const { return static_cast<int32_t>(code); }

std::string PrimitiveType::to_string() const { return TYPE_NAMES.at(code); }

void PrimitiveType::pack(BufferBuilder& buffer) const
{
  buffer.pack<int32_t>(static_cast<int32_t>(code));
}

bool PrimitiveType::equal(const Type& other) const { return code == other.code; }

ExtensionType::ExtensionType(int32_t uid, Type::Code code) : Type(code), uid_(uid) {}

FixedArrayType::FixedArrayType(int32_t uid,
                               std::shared_ptr<Type> element_type,
                               uint32_t N) noexcept(false)
  : ExtensionType(uid, Type::Code::FIXED_ARRAY),
    element_type_(std::move(element_type)),
    N_(N),
    size_(element_type_->size() * N)
{
  if (element_type_->variable_size()) throw std::invalid_argument(_VARIABLE_SIZE_ERROR_MESSAGE);
}

std::string FixedArrayType::to_string() const
{
  std::stringstream ss;
  ss << element_type_->to_string() << "[" << N_ << "]";
  return std::move(ss).str();
}

void FixedArrayType::pack(BufferBuilder& buffer) const
{
  buffer.pack<int32_t>(static_cast<int32_t>(code));
  buffer.pack<uint32_t>(uid_);
  buffer.pack<uint32_t>(N_);
  element_type_->pack(buffer);
}

const FixedArrayType& FixedArrayType::as_fixed_array_type() const { return *this; }

bool FixedArrayType::equal(const Type& other) const
{
  if (code != other.code) return false;
  auto& casted = static_cast<const FixedArrayType&>(other);

#ifdef DEBUG_LEGATE
  // Do a structural check in debug mode
  return uid_ == casted.uid_ && N_ == casted.N_ && element_type_ == casted.element_type_;
#else
  // Each type is uniquely identified by the uid, so it's sufficient to compare between uids
  return uid_ == casted.uid_;
#endif
}

StructType::StructType(int32_t uid,
                       std::vector<std::shared_ptr<Type>>&& field_types,
                       bool align) noexcept(false)
  : ExtensionType(uid, Type::Code::STRUCT),
    aligned_(align),
    alignment_(1),
    size_(0),
    field_types_(std::move(field_types))
{
  offsets_.reserve(field_types_.size());
  if (aligned_) {
    static constexpr auto align_offset = [](uint32_t offset, uint32_t align) {
      return (offset + (align - 1)) & -align;
    };

    for (auto& field_type : field_types_) {
      if (field_type->variable_size()) throw std::invalid_argument(_VARIABLE_SIZE_ERROR_MESSAGE);
      uint32_t _my_align = field_type->alignment();
      alignment_         = std::max(_my_align, alignment_);

      uint32_t offset = align_offset(size_, _my_align);
      offsets_.push_back(offset);
      size_ = offset + field_type->size();
    }
    size_ = align_offset(size_, alignment_);
  } else {
    for (auto& field_type : field_types_) {
      if (field_type->variable_size()) throw std::invalid_argument(_VARIABLE_SIZE_ERROR_MESSAGE);
      offsets_.push_back(size_);
      size_ += field_type->size();
    }
  }
}

std::string StructType::to_string() const
{
  std::stringstream ss;
  ss << "{";
  for (uint32_t idx = 0; idx < field_types_.size(); ++idx) {
    if (idx > 0) ss << ",";
    ss << field_types_.at(idx)->to_string() << ":" << offsets_.at(idx);
  }
  ss << "}";
  return std::move(ss).str();
}

void StructType::pack(BufferBuilder& buffer) const
{
  buffer.pack<int32_t>(static_cast<int32_t>(code));
  buffer.pack<uint32_t>(uid_);
  buffer.pack<uint32_t>(field_types_.size());
  for (auto& field_type : field_types_) field_type->pack(buffer);
  buffer.pack<bool>(aligned_);
}

const StructType& StructType::as_struct_type() const { return *this; }

bool StructType::equal(const Type& other) const
{
  if (code != other.code) return false;
  auto& casted = static_cast<const StructType&>(other);

#ifdef DEBUG_LEGATE
  // Do a structural check in debug mode
  if (uid_ != casted.uid_) return false;
  uint32_t nf = num_fields();
  if (nf != casted.num_fields()) return false;
  for (uint32_t idx = 0; idx < nf; ++idx)
    if (field_type(idx) != casted.field_type(idx)) return false;
  return true;
#else
  // Each type is uniquely identified by the uid, so it's sufficient to compare between uids
  return uid_ == casted.uid_;
#endif
}

std::shared_ptr<Type> StructType::field_type(uint32_t field_idx) const
{
  return field_types_.at(field_idx);
}

StringType::StringType() : Type(Type::Code::STRING) {}

int32_t StringType::uid() const { return static_cast<int32_t>(code); }

std::string StringType::to_string() const { return "string"; }

void StringType::pack(BufferBuilder& buffer) const
{
  buffer.pack<int32_t>(static_cast<int32_t>(code));
}

bool StringType::equal(const Type& other) const { return code == other.code; }

std::shared_ptr<Type> primitive_type(Type::Code code)
{
  return std::make_shared<PrimitiveType>(code);
}

std::shared_ptr<Type> string_type()
{
  static auto type = std::make_shared<StringType>();
  return type;
}

std::shared_ptr<Type> fixed_array_type(std::shared_ptr<Type> element_type,
                                       uint32_t N) noexcept(false)
{
  // We use UIDs of the following format for "common" fixed array types
  //    1B            1B
  // +--------+-------------------+
  // | length | element type code |
  // +--------+-------------------+
  int32_t uid = [&N](const Type& elem_type) {
    if (!elem_type.is_primitive() || N > 0xFFU) return Runtime::get_runtime()->get_type_uid();
    return static_cast<int32_t>(elem_type.code) | N << 8;
  }(*element_type);
  return std::make_shared<FixedArrayType>(uid, std::move(element_type), N);
}

std::shared_ptr<Type> struct_type(const std::vector<std::shared_ptr<Type>>& field_types,
                                  bool align) noexcept(false)
{
  return std::make_shared<StructType>(
    Runtime::get_runtime()->get_type_uid(), std::vector<std::shared_ptr<Type>>(field_types), align);
}

std::shared_ptr<Type> struct_type(std::vector<std::shared_ptr<Type>>&& field_types,
                                  bool align) noexcept(false)
{
  return std::make_shared<StructType>(
    Runtime::get_runtime()->get_type_uid(), std::move(field_types), align);
}

}  // namespace legate::detail
