/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <legion/legion_config.h>

#include <iosfwd>
#include <string>
#include <type_traits>
#include <vector>

/**
 * @file
 * @brief Legate type system
 */

namespace legate::detail {
class Type;
}  // namespace legate::detail

namespace legate {

/**
 * @addtogroup types
 * @{
 */

class FixedArrayType;
class ListType;
class StructType;

// Silence warnings here since the range is determined by whatever Legion sets, so we do not
// want to inadvertently muck that up.
// NOLINTBEGIN(performance-enum-size)
/**
 * @brief Enum for reduction operator kinds
 */
enum class ReductionOpKind : std::int32_t {
  ADD = LEGION_REDOP_KIND_SUM,  /*!< Addition */
  MUL = LEGION_REDOP_KIND_PROD, /*!< Multiplication */
  MAX = LEGION_REDOP_KIND_MAX,  /*!< Binary maximum operator */
  MIN = LEGION_REDOP_KIND_MIN,  /*!< Binary minimum operator */
  OR  = LEGION_REDOP_KIND_OR,   /*!< Bitwise OR */
  AND = LEGION_REDOP_KIND_AND,  /*!< Bitwse AND */
  XOR = LEGION_REDOP_KIND_XOR,  /*!< Bitwas XOR */
};
// NOLINTEND(performance-enum-size)

/**
 * @brief A base class for data type metadata
 */
class Type {
 public:
  // We silence performance-enum-size for the same reason we silence it for
  // ReductionOpKind. Also silence readability-enum-initial-value because we need the first few
  // values to map to Legion kinds exactly, but we don't care what the others are.
  // NOLINTBEGIN(performance-enum-size, readability-enum-initial-value)
  /**
   * @brief Enum for type codes
   */
  enum class Code : std::int32_t {
    BOOL       = LEGION_TYPE_BOOL,       /*!< Boolean type */
    INT8       = LEGION_TYPE_INT8,       /*!< 8-bit signed integer type */
    INT16      = LEGION_TYPE_INT16,      /*!< 16-bit signed integer type */
    INT32      = LEGION_TYPE_INT32,      /*!< 32-bit signed integer type */
    INT64      = LEGION_TYPE_INT64,      /*!< 64-bit signed integer type */
    UINT8      = LEGION_TYPE_UINT8,      /*!< 8-bit unsigned integer type */
    UINT16     = LEGION_TYPE_UINT16,     /*!< 16-bit unsigned integer type */
    UINT32     = LEGION_TYPE_UINT32,     /*!< 32-bit unsigned integer type */
    UINT64     = LEGION_TYPE_UINT64,     /*!< 64-bit unsigned integer type */
    FLOAT16    = LEGION_TYPE_FLOAT16,    /*!< Half-precision floating point type */
    FLOAT32    = LEGION_TYPE_FLOAT32,    /*!< Single-precision floating point type */
    FLOAT64    = LEGION_TYPE_FLOAT64,    /*!< Double-precision floating point type */
    COMPLEX64  = LEGION_TYPE_COMPLEX64,  /*!< Single-precision complex type */
    COMPLEX128 = LEGION_TYPE_COMPLEX128, /*!< Double-precision complex type */
    NIL,                                 /*!< Null type */
    BINARY,                              /*!< Opaque binary type */
    FIXED_ARRAY,                         /*!< Fixed-size array type */
    STRUCT,                              /*!< Struct type */
    STRING,                              /*!< String type */
    LIST,                                /*!< List type */
  };
  // NOLINTEND(performance-enum-size, readability-enum-initial-value)

  /**
   * @brief Code of the type
   *
   * @return Type code
   */
  [[nodiscard]] Code code() const;
  /**
   * @brief Size of the data type in bytes
   *
   * @return Data type size in bytes
   */
  [[nodiscard]] std::uint32_t size() const;
  /**
   * @brief Alignment of the type
   *
   * @return Alignment in bytes
   */
  [[nodiscard]] std::uint32_t alignment() const;
  /**
   * @brief Unique ID of the data type
   *
   * @return Unique ID
   */
  [[nodiscard]] std::uint32_t uid() const;
  /**
   * @brief Indicates whether the data type is of variable size elements
   *
   * @return true Elements can be variable size
   * @return false Elements have fixed size
   */
  [[nodiscard]] bool variable_size() const;
  /**
   * @brief Converts the data type into a string
   *
   * @return A string of the data type
   */
  [[nodiscard]] std::string to_string() const;
  /**
   * @brief Indicates whether the type is a primitive type
   *
   * @return true If the type is a primitive type
   * @return false Otherwise
   */
  [[nodiscard]] bool is_primitive() const;
  /**
   * @brief Dynamically casts the type into a fixed size array type.
   *
   * If the type is not a fixed size array type, an exception will be raised.
   *
   * @return Type object
   */
  [[nodiscard]] FixedArrayType as_fixed_array_type() const;
  /**
   * @brief Dynamically casts the type into a struct type.
   *
   * If the type is not a struct type, an exception will be raised.
   *
   * @return Type object
   */
  [[nodiscard]] StructType as_struct_type() const;
  /**
   * @brief Dynamically casts the type into a struct type.
   *
   * If the type is not a struct type, an exception will be raised.
   *
   * @return Type object
   */
  [[nodiscard]] ListType as_list_type() const;
  /**
   * @brief Records a reduction operator.
   *
   * The global ID of the reduction operator is issued when that operator is registered
   * to the runtime.
   *
   * @param op_kind Reduction operator kind
   * @param global_op_id Global reduction operator ID
   */
  void record_reduction_operator(std::int32_t op_kind, GlobalRedopID global_op_id) const;
  /**
   * @brief Records a reduction operator.
   *
   * The global ID of the reduction operator is issued when that operator is registered
   * to the runtime.
   *
   * @param op_kind Reduction operator kind
   * @param global_op_id Global reduction operator ID
   */
  void record_reduction_operator(ReductionOpKind op_kind, GlobalRedopID global_op_id) const;
  /**
   * @brief Finds the global operator ID for a given reduction operator kind.
   *
   * Raises an exception if no reduction operator has been registered for the kind.
   *
   * @param op_kind Reduction operator kind
   *
   * @return Global reduction operator ID
   */
  [[nodiscard]] GlobalRedopID find_reduction_operator(std::int32_t op_kind) const;
  /**
   * @brief Finds the global operator ID for a given reduction operator kind.
   *
   * Raises an exception if no reduction operator has been registered for the kind.
   *
   * @param op_kind Reduction operator kind
   *
   * @return Global reduction operator ID
   */
  [[nodiscard]] GlobalRedopID find_reduction_operator(ReductionOpKind op_kind) const;
  /**
   * @brief Equality check between types
   *
   * Note that type checks are name-based; two isomorphic fixed-size array types are considered
   * different if their uids are different (the same applies to struct types).
   *
   * @param other Type to compare
   *
   * @return true Types are equal
   * @return false Types are different
   */
  bool operator==(const Type& other) const;
  bool operator!=(const Type& other) const;

  Type() = LEGATE_DEFAULT_WHEN_CYTHON;

  Type(const Type&)                = default;
  Type(Type&&) noexcept            = default;
  Type& operator=(const Type&)     = default;
  Type& operator=(Type&&) noexcept = default;
  virtual ~Type()                  = default;

  explicit Type(InternalSharedPtr<detail::Type> impl);

  [[nodiscard]] const SharedPtr<detail::Type>& impl() const;

 protected:
  SharedPtr<detail::Type> impl_{};
};

/**
 * @brief A class for fixed-size array data types
 */
class FixedArrayType : public Type {
 public:
  FixedArrayType() = LEGATE_DEFAULT_WHEN_CYTHON;

  /**
   * @brief Returns the number of elements
   *
   * @return Number of elements
   */
  [[nodiscard]] std::uint32_t num_elements() const;
  /**
   * @brief Returns the element type
   *
   * @return Element type
   */
  [[nodiscard]] Type element_type() const;

  explicit FixedArrayType(InternalSharedPtr<detail::Type> type);
};

/**
 * @brief A class for struct data types
 */
class StructType : public Type {
 public:
  StructType() = LEGATE_DEFAULT_WHEN_CYTHON;

  /**
   * @brief Returns the number of fields
   *
   * @return Number of fields
   */
  [[nodiscard]] std::uint32_t num_fields() const;
  /**
   * @brief Returns the element type
   *
   * @param field_idx Field index. Must be within the range
   *
   * @return Element type
   */
  [[nodiscard]] Type field_type(std::uint32_t field_idx) const;
  /**
   * @brief Indicates whether the fields are aligned
   *
   * @return true Fields are aligned
   * @return false Fields are compact
   */
  [[nodiscard]] bool aligned() const;
  /**
   * @brief Returns offsets to fields
   *
   * @return Field offsets in a vector
   */
  [[nodiscard]] std::vector<std::uint32_t> offsets() const;

  explicit StructType(InternalSharedPtr<detail::Type> type);
};

/**
 * @brief A class for list types
 */
class ListType : public Type {
 public:
  ListType() = LEGATE_DEFAULT_WHEN_CYTHON;

  /**
   * @brief Returns the element type
   *
   * @return Element type
   */
  [[nodiscard]] Type element_type() const;

  explicit ListType(InternalSharedPtr<detail::Type> type);
};

/**
 * @brief Creates a metadata object for a primitive type
 *
 * @param code Type code
 *
 * @return Type object
 */
[[nodiscard]] Type primitive_type(Type::Code code);

/**
 * @brief Creates a metadata object for the string type
 *
 * @return Type object
 */
[[nodiscard]] Type string_type();

/**
 * @brief Creates an opaque binary type of a given size
 *
 * @param size Element size
 *
 * @return Type object
 */
[[nodiscard]] Type binary_type(std::uint32_t size);

/**
 * @brief Creates a metadata object for a fixed-size array type
 *
 * @param element_type Type of the array elements
 * @param N Size of the array
 *
 * @return FixedArrayType object
 */
[[nodiscard]] FixedArrayType fixed_array_type(const Type& element_type, std::uint32_t N);

/**
 * @brief Creates a metadata object for a struct type
 *
 * @param field_types A vector of field types
 * @param align If true, fields in the struct are aligned
 *
 * @return StructType object
 */
[[nodiscard]] StructType struct_type(const std::vector<Type>& field_types, bool align = true);

/**
 * @brief Creates a metadata object for a list type
 *
 * @param element_type Type of the list elements
 *
 * @return ListType object
 */
[[nodiscard]] ListType list_type(const Type& element_type);

/**
 * @brief Creates a metadata object for a struct type
 *
 * @param align If true, fields in the struct are aligned
 * @param field_types Field types
 *
 * @return StructType object
 */
template <typename... Args>
[[nodiscard]] std::enable_if_t<std::conjunction_v<std::is_convertible<std::decay_t<Args>, Type>...>,
                               StructType>
struct_type(bool align, Args&&... field_types);

std::ostream& operator<<(std::ostream&, const Type::Code&);

std::ostream& operator<<(std::ostream&, const Type&);

/**
 * @brief Creates a boolean type
 *
 * @return Type object
 */
[[nodiscard]] Type bool_();  // NOLINT(readability-identifier-naming)

/**
 * @brief Creates a 8-bit signed integer type
 *
 * @return Type object
 */
[[nodiscard]] Type int8();

/**
 * @brief Creates a 16-bit signed integer type
 *
 * @return Type object
 */
[[nodiscard]] Type int16();

/**
 * @brief Creates a 32-bit signed integer type
 *
 * @return Type object
 */
[[nodiscard]] Type int32();

/**
 * @brief Creates a 64-bit signed integer type
 *
 * @return Type object
 */
[[nodiscard]] Type int64();

/**
 * @brief Creates a 8-bit unsigned integer type
 *
 * @return Type object
 */
[[nodiscard]] Type uint8();

/**
 * @brief Creates a 16-bit unsigned integer type
 *
 * @return Type object
 */
[[nodiscard]] Type uint16();

/**
 * @brief Creates a 32-bit unsigned integer type
 *
 * @return Type object
 */
[[nodiscard]] Type uint32();

/**
 * @brief Creates a 64-bit unsigned integer type
 *
 * @return Type object
 */
[[nodiscard]] Type uint64();

/**
 * @brief Creates a half-precision floating point type
 *
 * @return Type object
 */
[[nodiscard]] Type float16();

/**
 * @brief Creates a single-precision floating point type
 *
 * @return Type object
 */
[[nodiscard]] Type float32();

/**
 * @brief Creates a double-precision floating point type
 *
 * @return Type object
 */
[[nodiscard]] Type float64();

/**
 * @brief Creates a single-precision complex number type
 *
 * @return Type object
 */
[[nodiscard]] Type complex64();

/**
 * @brief Creates a double-precision complex number type
 *
 * @return Type object
 */
[[nodiscard]] Type complex128();

/**
 * @brief Creates a point type
 *
 * @param ndim Number of dimensions
 *
 * @return FixedArrayType object
 */
[[nodiscard]] FixedArrayType point_type(std::uint32_t ndim);

/**
 * @brief Creates a rect type
 *
 * @param ndim Number of dimensions
 *
 * @return StructType object
 */
[[nodiscard]] StructType rect_type(std::uint32_t ndim);

/**
 * @brief Creates a null type
 *
 * @return Type object
 */
[[nodiscard]] Type null_type();

/**
 * @brief Checks if the type is a point type
 *
 * @param type Type to check
 *
 * @return true If the `type` is a point type
 * @return false Otherwise
 */
[[nodiscard]] bool is_point_type(const Type& type);

/**
 * @brief Checks if the type is a point type of the given dimensionality
 *
 * @param type Type to check
 * @param ndim Number of dimensions the point type should have
 *
 * @return true If the `type` is a point type
 * @return false Otherwise
 */
[[nodiscard]] bool is_point_type(const Type& type, std::uint32_t ndim);

/**
 * @brief Returns the number of dimensions of a given point type
 *
 * @param type Point type
 *
 * @return Number of dimensions
 *
 * @throw std::invalid_argument IF the type is not a point type
 */
[[nodiscard]] std::int32_t ndim_point_type(const Type& type);

/**
 * @brief Checks if the type is a rect type
 *
 * @param type Type to check
 *
 * @return true If the `type` is a rect type
 * @return false Otherwise
 */
[[nodiscard]] bool is_rect_type(const Type& type);

/**
 * @brief Checks if the type is a rect type of the given dimensionality
 *
 * @param type Type to check
 * @param ndim Number of dimensions the rect type should have
 *
 * @return true If the `type` is a rect type
 * @return false Otherwise
 */
[[nodiscard]] bool is_rect_type(const Type& type, std::uint32_t ndim);

/**
 * @brief Returns the number of dimensions of a given rect type
 *
 * @param type Rect type
 *
 * @return Number of dimensions
 *
 * @throw std::invalid_argument IF the type is not a rect type
 */
[[nodiscard]] std::int32_t ndim_rect_type(const Type& type);

/** @} */

}  // namespace legate

#include <legate/type/types.inl>
