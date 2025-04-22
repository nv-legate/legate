/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/type/detail/types.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/detail/zstring_view.h>

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <optional>
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

[[nodiscard]] ZStringView TYPE_NAME(Type::Code code)
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
  throw TracedException<std::invalid_argument>{"invalid type code"};
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
      throw TracedException<StaticDeterminationError>{
        fmt::format("Cannot statically determine size of non-integral type: {}", code)};
    }
  };
  throw TracedException<std::invalid_argument>{"invalid type code"};
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
      throw TracedException<StaticDeterminationError>{
        fmt::format("Cannot statically determine alignment of non-integral type: {}", code)};
    }
  };
  throw TracedException<std::invalid_argument>{"invalid type code"};
}

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

[[nodiscard]] std::uint32_t get_next_uid()
{
  static std::atomic<std::uint32_t> next_uid = BASE_CUSTOM_TYPE_UID;

  return next_uid.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

std::uint32_t Type::size() const
{
  throw TracedException<std::invalid_argument>{"Size of a variable size type is undefined"};
}

void Type::record_reduction_operator(std::int32_t op_kind, GlobalRedopID global_op_id) const
{
  detail::Runtime::get_runtime()->record_reduction_operator(uid(), op_kind, global_op_id);
}

GlobalRedopID Type::find_reduction_operator(std::int32_t op_kind) const
{
  return detail::Runtime::get_runtime()->find_reduction_operator(uid(), op_kind);
}

GlobalRedopID Type::find_reduction_operator(ReductionOpKind op_kind) const
{
  return find_reduction_operator(static_cast<std::int32_t>(op_kind));
}

PrimitiveType::PrimitiveType(Code type_code)
  : Type{type_code}, size_{SIZEOF(type_code)}, alignment_{ALIGNOF(type_code)}
{
}

std::string PrimitiveType::to_string() const { return TYPE_NAME(code).to_string(); }

void PrimitiveType::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(static_cast<std::int32_t>(code));
}

std::string BinaryType::to_string() const { return fmt::format("binary({})", size()); }

void BinaryType::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(static_cast<std::int32_t>(code));
  buffer.pack<std::uint32_t>(size());
}

FixedArrayType::FixedArrayType(std::uint32_t uid,
                               InternalSharedPtr<Type> element_type,
                               std::uint32_t N)
  : ExtensionType{uid, Type::Code::FIXED_ARRAY},
    element_type_{std::move(element_type)},
    N_{N},
    size_{this->element_type()->size() * N}
{
  if (this->element_type()->variable_size()) {
    throw TracedException<std::invalid_argument>{"Variable-size element type cannot be used"};
  }
}

std::string FixedArrayType::to_string() const
{
  return fmt::format("{}[{}]", *element_type(), num_elements());
}

void FixedArrayType::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(static_cast<std::int32_t>(code));
  buffer.pack<std::uint32_t>(uid());
  buffer.pack<std::uint32_t>(num_elements());
  element_type_->pack(buffer);
}

bool FixedArrayType::operator==(const Type& other) const
{
  if (code != other.code) {
    return false;
  }

  const auto& casted = static_cast<const FixedArrayType&>(other);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    // Do a structural check in debug mode
    return uid() == casted.uid() && num_elements() == casted.num_elements() &&
           element_type() == casted.element_type();
  }
  // Each type is uniquely identified by the uid, so it's sufficient to compare between uids
  return uid() == casted.uid();
}

StructType::StructType(std::uint32_t uid,
                       std::vector<InternalSharedPtr<Type>>&& field_types,
                       bool align)
  : ExtensionType{uid, Type::Code::STRUCT},
    aligned_{align},
    alignment_{1},
    field_types_{std::move(field_types)}
{
  if (std::any_of(this->field_types().begin(), this->field_types().end(), [](const auto& ty) {
        return ty->variable_size();
      })) {
    throw TracedException<std::runtime_error>{"Struct types can't have a variable size field"};
  }
  if (this->field_types().empty()) {
    throw TracedException<std::invalid_argument>{"Struct types must have at least one field"};
  }

  offsets_.reserve(this->field_types().size());
  if (aligned()) {
    static constexpr auto align_offset = [](std::uint32_t offset, std::uint32_t alignment) {
      return (offset + (alignment - 1)) & -alignment;
    };

    for (auto&& field_type : this->field_types()) {
      const auto my_align = field_type->alignment();
      alignment_          = std::max(my_align, alignment_);

      const auto offset = align_offset(size_, my_align);

      offsets_.push_back(offset);
      size_ = offset + field_type->size();
    }
    size_ = align_offset(size_, alignment_);
  } else {
    for (auto&& field_type : this->field_types()) {
      offsets_.push_back(size());
      size_ += field_type->size();
    }
  }
}

std::string StructType::to_string() const
{
  std::string result = "{";

  LEGATE_ASSERT(field_types().size() == offsets().size());
  for (auto&& [idx, rest] : enumerate(zip_equal(field_types(), offsets()))) {
    auto&& [field_type, offset] = rest;

    if (idx > 0) {
      result += ",";
    }
    fmt::format_to(std::back_inserter(result), "{}:{}", *field_type, offset);
  }
  result += "}";
  return result;
}

void StructType::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(static_cast<std::int32_t>(code));
  buffer.pack<std::uint32_t>(uid());
  buffer.pack<std::uint32_t>(static_cast<std::uint32_t>(field_types().size()));
  for (auto&& field_type : field_types()) {
    field_type->pack(buffer);
  }
  buffer.pack<bool>(aligned());
}

bool StructType::operator==(const Type& other) const
{
  if (code != other.code) {
    return false;
  }

  const auto& casted = static_cast<const StructType&>(other);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    // Do a structural check in debug mode
    return uid() == casted.uid() && field_types() == casted.field_types();
  }
  // Each type is uniquely identified by the uid, so it's sufficient to compare between uids
  return uid() == casted.uid();
}

const InternalSharedPtr<Type>& StructType::field_type(std::uint32_t field_idx) const
{
  return field_types().at(field_idx);
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
    std::string result;

    try {
      fmt::format_to(std::back_inserter(result), "{}", code);
    } catch (const std::invalid_argument&) {
      fmt::format_to(std::back_inserter(result), "<unknown type code: {}>", to_underlying(code));
    }
    fmt::format_to(std::back_inserter(result), " is not a valid type code for a primitive type");
    throw TracedException<std::invalid_argument>{std::move(result)};
  }

  const auto [it, inserted] = cache.try_emplace(code);

  if (inserted) {
    it->second = make_internal_shared<PrimitiveType>(code);
  }
  return it->second;
}

ListType::ListType(std::uint32_t uid, InternalSharedPtr<Type> element_type)
  : ExtensionType{uid, Type::Code::LIST}, element_type_{std::move(element_type)}
{
  if (this->element_type()->variable_size()) {
    throw TracedException<std::runtime_error>{"Nested variable size types are not implemented yet"};
  }
}

std::string ListType::to_string() const { return fmt::format("list({})", *element_type()); }

void ListType::pack(BufferBuilder& buffer) const
{
  buffer.pack<std::int32_t>(static_cast<std::int32_t>(code));
  buffer.pack<std::uint32_t>(uid());
  element_type()->pack(buffer);
}

bool ListType::operator==(const Type& other) const
{
  if (code != other.code) {
    return false;
  }

  const auto& casted = static_cast<const ListType&>(other);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    // Do a structural check in debug mode
    return uid() == casted.uid() && *element_type() == *casted.element_type();
  }
  // Each type is uniquely identified by the uid, so it's sufficient to compare between uids
  return uid() == casted.uid();
}

InternalSharedPtr<Type> string_type()
{
  static const auto type = make_internal_shared<StringType>();
  return type;
}

InternalSharedPtr<Type> binary_type(std::uint32_t size)
{
  if (size > MAX_BINARY_TYPE_SIZE) {
    throw TracedException<std::out_of_range>{
      fmt::format("Maximum size for opaque binary types is {}", MAX_BINARY_TYPE_SIZE)};
  }
  // size + 1 to account for size = 0
  const auto uid = static_cast<std::int32_t>(Type::Code::BINARY) | ((size + 1) << TYPE_CODE_OFFSET);

  return make_internal_shared<BinaryType>(uid, size);
}

InternalSharedPtr<FixedArrayType> fixed_array_type(InternalSharedPtr<Type> element_type,
                                                   std::uint32_t N)
{
  // We use UIDs of the following format for "common" fixed array types
  //    1B            1B
  // +--------+-------------------+
  // | length | element type code |
  // +--------+-------------------+
  const auto uid = [&] {
    constexpr auto MAX_ELEMENTS = 0xFFU;

    if (!element_type->is_primitive() || N > MAX_ELEMENTS) {
      return get_next_uid();
    }
    return static_cast<std::uint32_t>(to_underlying(element_type->code)) | (N << TYPE_CODE_OFFSET);
  }();
  return make_internal_shared<FixedArrayType>(uid, std::move(element_type), N);
}

InternalSharedPtr<StructType> struct_type(std::vector<InternalSharedPtr<Type>> field_types,
                                          bool align)
{
  return make_internal_shared<StructType>(get_next_uid(), std::move(field_types), align);
}

InternalSharedPtr<ListType> list_type(InternalSharedPtr<Type> element_type)
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

InternalSharedPtr<FixedArrayType> point_type(std::uint32_t ndim)
{
  static std::optional<InternalSharedPtr<FixedArrayType>> cache[LEGATE_MAX_DIM + 1];

  if (0 == ndim || ndim > LEGATE_MAX_DIM) {
    throw TracedException<std::out_of_range>{
      fmt::format("{} is not a supported number of dimensions", ndim)};
  }

  auto& opt = cache[ndim];

  if (!opt.has_value()) {
    opt = make_internal_shared<detail::FixedArrayType>(BASE_POINT_TYPE_UID + ndim, int64(), ndim);
  }
  return *opt;
}

InternalSharedPtr<StructType> rect_type(std::uint32_t ndim)
{
  static std::optional<InternalSharedPtr<StructType>> cache[LEGATE_MAX_DIM + 1];

  if (0 == ndim || ndim > LEGATE_MAX_DIM) {
    throw TracedException<std::out_of_range>{
      fmt::format("{} is not a supported number of dimensions", ndim)};
  }

  auto& opt = cache[ndim];

  if (!opt.has_value()) {
    auto pt_type = point_type(ndim);
    std::vector<InternalSharedPtr<detail::Type>> field_types{pt_type, pt_type};

    opt = make_internal_shared<detail::StructType>(
      BASE_RECT_TYPE_UID + ndim, std::move(field_types), true /*align*/);
  }
  return *opt;
}

InternalSharedPtr<Type> null_type()
{
  static const auto result = detail::primitive_type(Type::Code::NIL);
  return result;
}

InternalSharedPtr<Type> domain_type()
{
  static const auto result = detail::binary_type(sizeof(Domain));
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
    throw TracedException<std::invalid_argument>{
      fmt::format("Expected a point type but got {}", *type)};
  }
  return type->code == Type::Code::INT64
           ? 1
           : static_cast<std::int32_t>(type->uid() - BASE_POINT_TYPE_UID);
}

bool is_rect_type(const InternalSharedPtr<Type>& type)
{
  const auto uid = type->uid();

  return uid > BASE_RECT_TYPE_UID && uid <= BASE_RECT_TYPE_UID + LEGATE_MAX_DIM;
}

bool is_rect_type(const InternalSharedPtr<Type>& type, std::uint32_t ndim)
{
  return type->uid() == BASE_RECT_TYPE_UID + ndim;
}

std::int32_t ndim_rect_type(const InternalSharedPtr<Type>& type)
{
  if (!is_rect_type(type)) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Expected a rect type but got {}", *type)};
  }
  return static_cast<std::int32_t>(type->uid() - BASE_RECT_TYPE_UID);
}

namespace {

std::int32_t statically_initialize_types() noexcept
{
  try {
    // The following functions cache type objects in function-local static variables to avoid
    // dynamic allocations for types that are used frequently in programs. Despite the C++ standard
    // guaranteeing that concurrent initializations for those cached objects are safe, these
    // functions aren't actually thread safe when their static variables are uninitialized, because
    // they internally call the `primitive_type()` function that doesn't guard its cache stored in
    // an `unordered_map`. A fix for cases like this usually involves adding a lock to the shared
    // data structures to protect it from concurrent accesses. In this case, however, we know that
    // the cache would be populated only by the primitive types and need to be initialized only
    // once, so we instead just initialize the caches by calling these functions statically when the
    // library gets loaded for the first time.
    static_cast<void>(bool_());
    static_cast<void>(int8());
    static_cast<void>(int16());
    static_cast<void>(int32());
    static_cast<void>(int64());
    static_cast<void>(uint8());
    static_cast<void>(uint16());
    static_cast<void>(uint32());
    static_cast<void>(uint64());
    static_cast<void>(float16());
    static_cast<void>(float32());
    static_cast<void>(float64());
    static_cast<void>(complex64());
    static_cast<void>(complex128());
    static_cast<void>(null_type());
  } catch (const std::exception& exn) {
    LEGATE_ABORT(exn.what());
  }
  return 0;
}

const std::int32_t STATICALLY_INITIALIZE_TYPES = statically_initialize_types();

}  // namespace

}  // namespace legate::detail

namespace fmt {

format_context::iterator formatter<legate::detail::Type::Code>::format(legate::detail::Type::Code a,
                                                                       format_context& ctx) const
{
  return formatter<legate::detail::ZStringView>::format(legate::detail::TYPE_NAME(a), ctx);
}

}  // namespace fmt
