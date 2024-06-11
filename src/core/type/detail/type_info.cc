/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/type/detail/type_info.h"

#include "core/runtime/detail/runtime.h"
#include "core/utilities/detail/buffer_builder.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace legate::detail {

namespace {

class StaticDeterminationError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

[[nodiscard]] std::string_view TYPE_NAME(Type::Code code)
{
  switch (code) {
    case Type::Code::BOOL: return "bool";
    case Type::Code::INT8: return "int8";
    case Type::Code::INT16: return "int16";
    case Type::Code::INT32: return "int32";
    case Type::Code::INT64: return "int64";
    case Type::Code::UINT8: return "uint8";
    case Type::Code::UINT16: return "uint16";
    case Type::Code::UINT32: return "uint32";
    case Type::Code::UINT64: return "uint64";
    case Type::Code::FLOAT16: return "float16";
    case Type::Code::FLOAT32: return "float32";
    case Type::Code::FLOAT64: return "float64";
    case Type::Code::COMPLEX64: return "complex64";
    case Type::Code::COMPLEX128: return "complex128";
    case Type::Code::STRING: return "string";
    case Type::Code::NIL: return "null_type";
    case Type::Code::BINARY: return "binary";
    case Type::Code::STRUCT: return "struct";
    case Type::Code::FIXED_ARRAY: return "fixed_array";
    case Type::Code::LIST: return "list";
  }
  throw std::invalid_argument{"invalid type code"};
}

[[nodiscard]] std::uint32_t SIZEOF(Type::Code code)
{
  switch (code) {
#define SIZEOF_TYPE_CODE(x) \
  case x: return sizeof(type_of_t<x>)

    SIZEOF_TYPE_CODE(Type::Code::BOOL);
    SIZEOF_TYPE_CODE(Type::Code::INT8);
    SIZEOF_TYPE_CODE(Type::Code::INT16);
    SIZEOF_TYPE_CODE(Type::Code::INT32);
    SIZEOF_TYPE_CODE(Type::Code::INT64);
    SIZEOF_TYPE_CODE(Type::Code::UINT8);
    SIZEOF_TYPE_CODE(Type::Code::UINT16);
    SIZEOF_TYPE_CODE(Type::Code::UINT32);
    SIZEOF_TYPE_CODE(Type::Code::UINT64);
    SIZEOF_TYPE_CODE(Type::Code::FLOAT16);
    SIZEOF_TYPE_CODE(Type::Code::FLOAT32);
    SIZEOF_TYPE_CODE(Type::Code::FLOAT64);
    SIZEOF_TYPE_CODE(Type::Code::COMPLEX64);
    SIZEOF_TYPE_CODE(Type::Code::COMPLEX128);

#undef SIZEOF_TYPE_CODE

    case Type::Code::NIL: return 0;
    case Type::Code::BINARY: [[fallthrough]];
    case Type::Code::STRUCT: [[fallthrough]];
    case Type::Code::FIXED_ARRAY: [[fallthrough]];
    case Type::Code::LIST: [[fallthrough]];
    case Type::Code::STRING: {
      std::stringstream ss;

      ss << "Cannot statically determine size of non-integral type: " << TYPE_NAME(code);
      throw StaticDeterminationError{std::move(ss).str()};
    }
  };
  throw std::invalid_argument{"invalid type code"};
}

[[nodiscard]] std::uint32_t ALIGNOF(Type::Code code)
{
  switch (code) {
#define ALIGNOF_TYPE_CODE(x) \
  case x: return alignof(type_of_t<x>)

    ALIGNOF_TYPE_CODE(Type::Code::BOOL);
    ALIGNOF_TYPE_CODE(Type::Code::INT8);
    ALIGNOF_TYPE_CODE(Type::Code::INT16);
    ALIGNOF_TYPE_CODE(Type::Code::INT32);
    ALIGNOF_TYPE_CODE(Type::Code::INT64);
    ALIGNOF_TYPE_CODE(Type::Code::UINT8);
    ALIGNOF_TYPE_CODE(Type::Code::UINT16);
    ALIGNOF_TYPE_CODE(Type::Code::UINT32);
    ALIGNOF_TYPE_CODE(Type::Code::UINT64);
    ALIGNOF_TYPE_CODE(Type::Code::FLOAT16);
    ALIGNOF_TYPE_CODE(Type::Code::FLOAT32);
    ALIGNOF_TYPE_CODE(Type::Code::FLOAT64);
    ALIGNOF_TYPE_CODE(Type::Code::COMPLEX64);
    ALIGNOF_TYPE_CODE(Type::Code::COMPLEX128);

#undef ALIGNOF_TYPE_CODE

    case Type::Code::NIL: return 0;
    case Type::Code::BINARY: [[fallthrough]];
    case Type::Code::STRUCT: [[fallthrough]];
    case Type::Code::FIXED_ARRAY: [[fallthrough]];
    case Type::Code::LIST: [[fallthrough]];
    case Type::Code::STRING: {
      std::stringstream ss;

      ss << "Cannot statically determine alingment of non-integral type: " << TYPE_NAME(code);
      throw StaticDeterminationError{std::move(ss).str()};
    }
  };
  throw std::invalid_argument{"invalid type code"};
}

constexpr const char* const VARIABLE_SIZE_ERROR_MESSAGE =
  "Variable-size element type cannot be used";

// Some notes about these magic numbers:
//
// The numbers are chosen such that UIDs of types are truly unique even in the presence of types
// with static UIDs derived from their type codes and sizes. Here's the list of static UIDs that
// each kind of types can take (dynamic UIDs generated off of _BASE_CUSTOM_TYPE_UID are unique by
// construction):
//
// * Primitive types: [0x00, 0x0E]
// * Binary types: [0x000001, 0x0FFFFF] <+> [0x0F]
// * Fixed-size array types: [0x01, 0xFF] <+> [0x00, 0x0E]
// * Point types: [_BASE_POINT_TYPE_UID + 1, _BASE_POINT_TYPE_UID + LEGATE_MAX_DIM]
// * Rect types: [_BASE_RECT_TYPE_UID + 1, _BASE_RECT_TYPE_UID + LEGATE_MAX_DIM]
//
// where the <+> operator is a pairwise concatenation
constexpr std::uint32_t TYPE_CODE_OFFSET     = 8;
constexpr std::uint32_t BASE_POINT_TYPE_UID  = 0x10000000;
constexpr std::uint32_t BASE_RECT_TYPE_UID   = BASE_POINT_TYPE_UID + LEGATE_MAX_DIM + 1;
constexpr std::uint32_t BASE_CUSTOM_TYPE_UID = BASE_RECT_TYPE_UID + LEGATE_MAX_DIM + 1;
// Last byte of a static UID is a type code
constexpr std::uint32_t MAX_BINARY_TYPE_SIZE = 0x0FFFFF00 >> TYPE_CODE_OFFSET;

std::uint32_t get_next_uid()
{
  static std::atomic<std::uint32_t> next_uid = BASE_CUSTOM_TYPE_UID;
  return next_uid++;
}

}  // namespace

std::uint32_t Type::size() const
{
  throw std::invalid_argument{"Size of a variable size type is undefined"};
  return {};
}

void Type::record_reduction_operator(std::int32_t op_kind, std::int32_t global_op_id) const
{
  detail::Runtime::get_runtime()->record_reduction_operator(uid(), op_kind, global_op_id);
}

std::int32_t Type::find_reduction_operator(std::int32_t op_kind) const
{
  return detail::Runtime::get_runtime()->find_reduction_operator(uid(), op_kind);
}

std::int32_t Type::find_reduction_operator(ReductionOpKind op_kind) const
{
  return find_reduction_operator(static_cast<std::int32_t>(op_kind));
}

bool Type::operator==(const Type& other) const { return equal(other); }

PrimitiveType::PrimitiveType(Code type_code)
  : Type{type_code}, size_{SIZEOF(type_code)}, alignment_{ALIGNOF(type_code)}
{
}

std::string PrimitiveType::to_string() const { return std::string{TYPE_NAME(code)}; }

void PrimitiveType::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(static_cast<std::int32_t>(code));
}

std::string BinaryType::to_string() const { return "binary(" + std::to_string(size_) + ")"; }

void BinaryType::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(static_cast<std::int32_t>(code));
  buffer.pack<std::uint32_t>(size_);
}

FixedArrayType::FixedArrayType(std::uint32_t uid,
                               InternalSharedPtr<Type> element_type,
                               std::uint32_t N)
  : ExtensionType{uid, Type::Code::FIXED_ARRAY},
    element_type_{std::move(element_type)},
    N_{N},
    size_{element_type_->size() * N}
{
  if (element_type_->variable_size()) {
    throw std::invalid_argument{VARIABLE_SIZE_ERROR_MESSAGE};
  }
}

std::string FixedArrayType::to_string() const
{
  std::stringstream ss;

  ss << element_type_->to_string() << "[" << N_ << "]";
  return std::move(ss).str();
}

void FixedArrayType::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(static_cast<std::int32_t>(code));
  buffer.pack<std::uint32_t>(uid_);
  buffer.pack<std::uint32_t>(N_);
  element_type_->pack(buffer);
}

bool FixedArrayType::equal(const Type& other) const
{
  if (code != other.code) {
    return false;
  }
  auto& casted = static_cast<const FixedArrayType&>(other);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    // Do a structural check in debug mode
    return uid_ == casted.uid_ && N_ == casted.N_ && element_type_ == casted.element_type_;
  }
  // Each type is uniquely identified by the uid, so it's sufficient to compare between uids
  return uid_ == casted.uid_;
}

StructType::StructType(std::uint32_t uid,
                       std::vector<InternalSharedPtr<Type>>&& field_types,
                       bool align)
  : ExtensionType{uid, Type::Code::STRUCT},
    aligned_{align},
    alignment_{1},
    field_types_{std::move(field_types)}
{
  if (std::any_of(
        field_types_.begin(), field_types_.end(), [](auto& ty) { return ty->variable_size(); })) {
    throw std::runtime_error{"Struct types can't have a variable size field"};
  }
  if (field_types_.empty()) {
    throw std::invalid_argument{"Struct types must have at least one field"};
  }

  offsets_.reserve(field_types_.size());
  if (aligned_) {
    static constexpr auto align_offset = [](std::uint32_t offset, std::uint32_t alignment) {
      return (offset + (alignment - 1)) & -alignment;
    };

    for (auto&& field_type : field_types_) {
      if (field_type->variable_size()) {
        throw std::invalid_argument{VARIABLE_SIZE_ERROR_MESSAGE};
      }
      const auto my_align = field_type->alignment();
      alignment_          = std::max(my_align, alignment_);

      const auto offset = align_offset(size_, my_align);
      offsets_.push_back(offset);
      size_ = offset + field_type->size();
    }
    size_ = align_offset(size_, alignment_);
  } else {
    for (auto&& field_type : field_types_) {
      if (field_type->variable_size()) {
        throw std::invalid_argument{VARIABLE_SIZE_ERROR_MESSAGE};
      }
      offsets_.push_back(size_);
      size_ += field_type->size();
    }
  }
}

std::string StructType::to_string() const
{
  std::stringstream ss;

  ss << "{";
  for (std::uint32_t idx = 0; idx < field_types_.size(); ++idx) {
    if (idx > 0) {
      ss << ",";
    }
    ss << field_types_.at(idx)->to_string() << ":" << offsets_.at(idx);
  }
  ss << "}";
  return std::move(ss).str();
}

void StructType::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(static_cast<std::int32_t>(code));
  buffer.pack<std::uint32_t>(uid_);
  buffer.pack<std::uint32_t>(static_cast<std::uint32_t>(field_types_.size()));
  for (auto&& field_type : field_types_) {
    field_type->pack(buffer);
  }
  buffer.pack<bool>(aligned_);
}

bool StructType::equal(const Type& other) const
{
  if (code != other.code) {
    return false;
  }
  auto& casted = static_cast<const StructType&>(other);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    // Do a structural check in debug mode
    if (uid_ != casted.uid_) {
      return false;
    }
    const auto nf = num_fields();
    if (nf != casted.num_fields()) {
      return false;
    }
    for (std::uint32_t idx = 0; idx < nf; ++idx) {
      if (field_type(idx) != casted.field_type(idx)) {
        return false;
      }
    }
    return true;
  }
  // Each type is uniquely identified by the uid, so it's sufficient to compare between uids
  return uid_ == casted.uid_;
}

const InternalSharedPtr<Type>& StructType::field_type(std::uint32_t field_idx) const
{
  return field_types_.at(field_idx);
}

void StringType::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(static_cast<std::int32_t>(code));
}

InternalSharedPtr<Type> primitive_type(Type::Code code)
{
  static std::unordered_map<Type::Code, InternalSharedPtr<Type>> cache{};

  try {
    static_cast<void>(SIZEOF(code));
  } catch (const StaticDeterminationError&) {
    std::stringstream ss;

    try {
      ss << TYPE_NAME(code);
    } catch (const std::invalid_argument&) {
      ss << "<unknown type code: " << traits::detail::to_underlying(code) << ">";
    }
    ss << " is not a valid type code for a primitive type";
    throw std::invalid_argument{std::move(ss).str()};
  }

  auto finder = cache.find(code);
  if (finder != cache.end()) {
    return finder->second;
  }
  return cache[code] = make_internal_shared<PrimitiveType>(code);
}

ListType::ListType(std::uint32_t uid, InternalSharedPtr<Type> element_type)
  : ExtensionType{uid, Type::Code::LIST}, element_type_{std::move(element_type)}
{
  if (element_type_->variable_size()) {
    throw std::runtime_error{"Nested variable size types are not implemented yet"};
  }
}

std::string ListType::to_string() const
{
  std::stringstream ss;

  ss << "list(" << element_type_->to_string() << ")";
  return std::move(ss).str();
}

void ListType::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(static_cast<std::int32_t>(code));
  buffer.pack<std::uint32_t>(uid_);
  element_type_->pack(buffer);
}

bool ListType::equal(const Type& other) const
{
  if (code != other.code) {
    return false;
  }
  auto& casted = static_cast<const ListType&>(other);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    // Do a structural check in debug mode
    return uid_ == casted.uid_ && element_type_ == casted.element_type_;
  }
  // Each type is uniquely identified by the uid, so it's sufficient to compare between uids
  return uid_ == casted.uid_;
}

InternalSharedPtr<Type> string_type()
{
  static const auto type = make_internal_shared<StringType>();
  return type;
}

InternalSharedPtr<Type> binary_type(std::uint32_t size)
{
  if (size == 0) {
    throw std::out_of_range{"Size for an opaque binary type must be greater than 0"};
  }
  if (size > MAX_BINARY_TYPE_SIZE) {
    throw std::out_of_range{"Maximum size for opaque binary types is " +
                            std::to_string(MAX_BINARY_TYPE_SIZE)};
  }
  auto uid = static_cast<std::int32_t>(Type::Code::BINARY) | (size << TYPE_CODE_OFFSET);
  return make_internal_shared<BinaryType>(uid, size);
}

InternalSharedPtr<Type> fixed_array_type(InternalSharedPtr<Type> element_type, std::uint32_t N)
{
  if (N == 0) {
    throw std::out_of_range{"Size of array must be greater than 0"};
  }
  // We use UIDs of the following format for "common" fixed array types
  //    1B            1B
  // +--------+-------------------+
  // | length | element type code |
  // +--------+-------------------+
  auto uid = [&N](const Type& elem_type) {
    constexpr auto MAX_ELEMENTS = 0xFFU;

    if (!elem_type.is_primitive() || N > MAX_ELEMENTS) {
      return get_next_uid();
    }
    return static_cast<std::int32_t>(elem_type.code) | (N << TYPE_CODE_OFFSET);
  }(*element_type);
  return make_internal_shared<FixedArrayType>(uid, std::move(element_type), N);
}

InternalSharedPtr<Type> struct_type(std::vector<InternalSharedPtr<Type>> field_types, bool align)
{
  return make_internal_shared<StructType>(get_next_uid(), std::move(field_types), align);
}

InternalSharedPtr<Type> list_type(InternalSharedPtr<Type> element_type)
{
  return make_internal_shared<ListType>(get_next_uid(), std::move(element_type));
}

InternalSharedPtr<Type> bool_()
{
  static const auto result = detail::primitive_type(Type::Code::BOOL);
  return result;
}

InternalSharedPtr<Type> int8()
{
  static const auto result = detail::primitive_type(Type::Code::INT8);
  return result;
}

InternalSharedPtr<Type> int16()
{
  static const auto result = detail::primitive_type(Type::Code::INT16);
  return result;
}

InternalSharedPtr<Type> int32()
{
  static const auto result = detail::primitive_type(Type::Code::INT32);
  return result;
}

InternalSharedPtr<Type> int64()
{
  static const auto result = detail::primitive_type(Type::Code::INT64);
  return result;
}

InternalSharedPtr<Type> uint8()
{
  static const auto result = detail::primitive_type(Type::Code::UINT8);
  return result;
}

InternalSharedPtr<Type> uint16()
{
  static const auto result = detail::primitive_type(Type::Code::UINT16);
  return result;
}

InternalSharedPtr<Type> uint32()
{
  static const auto result = detail::primitive_type(Type::Code::UINT32);
  return result;
}

InternalSharedPtr<Type> uint64()
{
  static const auto result = detail::primitive_type(Type::Code::UINT64);
  return result;
}

InternalSharedPtr<Type> float16()
{
  static const auto result = detail::primitive_type(Type::Code::FLOAT16);
  return result;
}

InternalSharedPtr<Type> float32()
{
  static const auto result = detail::primitive_type(Type::Code::FLOAT32);
  return result;
}

InternalSharedPtr<Type> float64()
{
  static const auto result = detail::primitive_type(Type::Code::FLOAT64);
  return result;
}

InternalSharedPtr<Type> complex64()
{
  static const auto result = detail::primitive_type(Type::Code::COMPLEX64);
  return result;
}

InternalSharedPtr<Type> complex128()
{
  static const auto result = detail::primitive_type(Type::Code::COMPLEX128);
  return result;
}

InternalSharedPtr<Type> point_type(std::uint32_t ndim)
{
  static InternalSharedPtr<Type> cache[LEGATE_MAX_DIM + 1];

  if (0 == ndim || ndim > LEGATE_MAX_DIM) {
    throw std::out_of_range{std::to_string(ndim) + " is not a supported number of dimensions"};
  }
  if (nullptr == cache[ndim]) {
    cache[ndim] =
      make_internal_shared<detail::FixedArrayType>(BASE_POINT_TYPE_UID + ndim, int64(), ndim);
  }
  return cache[ndim];
}

InternalSharedPtr<Type> rect_type(std::uint32_t ndim)
{
  static InternalSharedPtr<Type> cache[LEGATE_MAX_DIM + 1];

  if (0 == ndim || ndim > LEGATE_MAX_DIM) {
    throw std::out_of_range{std::to_string(ndim) + " is not a supported number of dimensions"};
  }

  if (nullptr == cache[ndim]) {
    auto pt_type = point_type(ndim);
    std::vector<InternalSharedPtr<detail::Type>> field_types{pt_type, pt_type};

    cache[ndim] = make_internal_shared<detail::StructType>(
      BASE_RECT_TYPE_UID + ndim, std::move(field_types), true /*align*/);
  }
  return cache[ndim];
}

InternalSharedPtr<Type> null_type()
{
  static const auto result = detail::primitive_type(Type::Code::NIL);
  return result;
}

InternalSharedPtr<Type> domain_type()
{
  static auto result = detail::binary_type(sizeof(Domain));
  return result;
}

bool is_point_type(const InternalSharedPtr<Type>& type)
{
  const auto uid = type->uid();
  return type->code == Type::Code::INT64 ||
         (uid > BASE_POINT_TYPE_UID && uid <= BASE_POINT_TYPE_UID + LEGATE_MAX_DIM);
}

bool is_point_type(const InternalSharedPtr<Type>& type, std::uint32_t ndim)
{
  return (ndim == 1 && type->code == Type::Code::INT64) ||
         type->uid() == BASE_POINT_TYPE_UID + ndim;
}

std::int32_t ndim_point_type(const InternalSharedPtr<Type>& type)
{
  if (!is_point_type(type)) {
    throw std::invalid_argument{"Expected a point type but got " + type->to_string()};
  }
  return type->code == Type::Code::INT64
           ? 1
           : static_cast<std::int32_t>(type->uid() - BASE_POINT_TYPE_UID);
}

bool is_rect_type(const InternalSharedPtr<Type>& type)
{
  auto uid = type->uid();
  return uid > BASE_RECT_TYPE_UID && uid <= BASE_RECT_TYPE_UID + LEGATE_MAX_DIM;
}

bool is_rect_type(const InternalSharedPtr<Type>& type, std::uint32_t ndim)
{
  return type->uid() == BASE_RECT_TYPE_UID + ndim;
}

std::int32_t ndim_rect_type(const InternalSharedPtr<Type>& type)
{
  if (!is_rect_type(type)) {
    throw std::invalid_argument{"Expected a rect type but got " + type->to_string()};
  }
  return static_cast<std::int32_t>(type->uid() - BASE_RECT_TYPE_UID);
}

}  // namespace legate::detail
