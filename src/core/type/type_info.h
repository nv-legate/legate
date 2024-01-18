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

#pragma once

#include "core/legate_c.h"
#include "core/utilities/shared_ptr.h"

#include <iosfwd>
#include <string>
#include <type_traits>
#include <vector>

/**
 * @file
 * @brief Legate type system
 */

/** @defgroup types Type system
 */

namespace legate::detail {
class Type;
}  // namespace legate::detail

namespace legate {

class FixedArrayType;
class ListType;
class StructType;

/**
 * @ingroup types
 * @brief Enum for reduction operator kinds
 */
enum class ReductionOpKind : int32_t {
  ADD = ADD_LT, /*!< Addition */
  SUB = SUB_LT, /*!< Subtraction */
  MUL = MUL_LT, /*!< Multiplication */
  DIV = DIV_LT, /*!< Division */
  MAX = MAX_LT, /*!< Binary maximum operator */
  MIN = MIN_LT, /*!< Binary minimum operator */
  OR  = OR_LT,  /*!< Bitwise OR */
  AND = AND_LT, /*!< Bitwse AND */
  XOR = XOR_LT, /*!< Bitwas XOR */
};

/**
 * @ingroup types
 * @brief A base class for data type metadata
 */
class Type {
 public:
  /**
   * @ingroup types
   * @brief Enum for type codes
   */
  enum class Code : int32_t {
    NIL         = NULL_LT,        /*!< Null type */
    BOOL        = BOOL_LT,        /*!< Boolean type */
    INT8        = INT8_LT,        /*!< 8-bit signed integer type */
    INT16       = INT16_LT,       /*!< 16-bit signed integer type */
    INT32       = INT32_LT,       /*!< 32-bit signed integer type */
    INT64       = INT64_LT,       /*!< 64-bit signed integer type */
    UINT8       = UINT8_LT,       /*!< 8-bit unsigned integer type */
    UINT16      = UINT16_LT,      /*!< 16-bit unsigned integer type */
    UINT32      = UINT32_LT,      /*!< 32-bit unsigned integer type */
    UINT64      = UINT64_LT,      /*!< 64-bit unsigned integer type */
    FLOAT16     = FLOAT16_LT,     /*!< Half-precision floating point type */
    FLOAT32     = FLOAT32_LT,     /*!< Single-precision floating point type */
    FLOAT64     = FLOAT64_LT,     /*!< Double-precision floating point type */
    COMPLEX64   = COMPLEX64_LT,   /*!< Single-precision complex type */
    COMPLEX128  = COMPLEX128_LT,  /*!< Double-precision complex type */
    BINARY      = BINARY_LT,      /*!< Opaque binary type */
    FIXED_ARRAY = FIXED_ARRAY_LT, /*!< Fixed-size array type */
    STRUCT      = STRUCT_LT,      /*!< Struct type */
    STRING      = STRING_LT,      /*!< String type */
    LIST        = LIST_LT,        /*!< List type */
  };

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
  [[nodiscard]] uint32_t size() const;
  /**
   * @brief Alignment of the type
   *
   * @return Alignment in bytes
   */
  [[nodiscard]] uint32_t alignment() const;
  /**
   * @brief Unique ID of the data type
   *
   * @return Unique ID
   */
  [[nodiscard]] uint32_t uid() const;
  /**
   * @brief Inidicates whether the data type is of varible size elements
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
  void record_reduction_operator(int32_t op_kind, int32_t global_op_id) const;
  /**
   * @brief Records a reduction operator.
   *
   * The global ID of the reduction operator is issued when that operator is registered
   * to the runtime.
   *
   * @param op_kind Reduction operator kind
   * @param global_op_id Global reduction operator ID
   */
  void record_reduction_operator(ReductionOpKind op_kind, int32_t global_op_id) const;
  /**
   * @brief Finds the global operator ID for a given reduction operator kind.
   *
   * Raises an exception if no reduction operator has been registered for the kind.
   *
   * @param op_kind Reduction operator kind
   *
   * @return Global reduction operator ID
   */
  [[nodiscard]] int32_t find_reduction_operator(int32_t op_kind) const;
  /**
   * @brief Finds the global operator ID for a given reduction operator kind.
   *
   * Raises an exception if no reduction operator has been registered for the kind.
   *
   * @param op_kind Reduction operator kind
   *
   * @return Global reduction operator ID
   */
  [[nodiscard]] int32_t find_reduction_operator(ReductionOpKind op_kind) const;
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

  Type();
  Type(const Type&)                = default;
  Type(Type&&) noexcept            = default;
  Type& operator=(const Type&)     = default;
  Type& operator=(Type&&) noexcept = default;

  virtual ~Type();

  explicit Type(InternalSharedPtr<detail::Type> impl);

  [[nodiscard]] const SharedPtr<detail::Type>& impl() const;

 protected:
  SharedPtr<detail::Type> impl_{};
};

/**
 * @ingroup types
 * @brief A class for fixed-size array data types
 */
class FixedArrayType : public Type {
 public:
  /**
   * @brief Returns the number of elements
   *
   * @return Number of elements
   */
  [[nodiscard]] uint32_t num_elements() const;
  /**
   * @brief Returns the element type
   *
   * @return Element type
   */
  [[nodiscard]] Type element_type() const;

 private:
  friend class Type;
  explicit FixedArrayType(InternalSharedPtr<detail::Type> type);
};

/**
 * @ingroup types
 * @brief A class for struct data types
 */
class StructType : public Type {
 public:
  /**
   * @brief Returns the number of fields
   *
   * @return Number of fields
   */
  [[nodiscard]] uint32_t num_fields() const;
  /**
   * @brief Returns the element type
   *
   * @param field_idx Field index. Must be within the range
   *
   * @return Element type
   */
  [[nodiscard]] Type field_type(uint32_t field_idx) const;
  /**
   * @brief Indiciates whether the fields are aligned
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
  [[nodiscard]] std::vector<uint32_t> offsets() const;

 private:
  friend class Type;
  explicit StructType(InternalSharedPtr<detail::Type> type);
};

/**
 * @ingroup types
 * @brief A class for list types
 */
class ListType : public Type {
 public:
  /**
   * @brief Returns the element type
   *
   * @return Element type
   */
  [[nodiscard]] Type element_type() const;

 private:
  friend class Type;
  explicit ListType(InternalSharedPtr<detail::Type> type);
};

/**
 * @ingroup types
 * @brief Creates a metadata object for a primitive type
 *
 * @param code Type code
 *
 * @return Type object
 */
[[nodiscard]] Type primitive_type(Type::Code code);

/**
 * @ingroup types
 * @brief Creates a metadata object for the string type
 *
 * @return Type object
 */
[[nodiscard]] Type string_type();

/**
 * @ingroup types
 * @brief Creates an opaque binary type of a given size
 *
 * @param size Element size
 *
 * @return Type object
 */
[[nodiscard]] Type binary_type(uint32_t size);

/**
 * @ingroup types
 * @brief Creates a metadata object for a fixed-size array type
 *
 * @param element_type Type of the array elements
 * @param N Size of the array
 *
 * @return Type object
 */
[[nodiscard]] Type fixed_array_type(const Type& element_type, uint32_t N);

/**
 * @ingroup types
 * @brief Creates a metadata object for a struct type
 *
 * @param field_types A vector of field types
 * @param align If true, fields in the struct are aligned
 *
 * @return Type object
 */
[[nodiscard]] Type struct_type(const std::vector<Type>& field_types, bool align = false);

/**
 * @ingroup types
 * @brief Creates a metadata object for a list type
 *
 * @param element_type Type of the list elements
 *
 * @return Type object
 */
[[nodiscard]] Type list_type(const Type& element_type);

/**
 * @ingroup types
 * @brief Creates a metadata object for a struct type
 *
 * @param align If true, fields in the struct are aligned
 * @param field_types Field types
 *
 * @return Type object
 */
template <typename... Args>
[[nodiscard]] std::enable_if_t<std::conjunction_v<std::is_same<std::decay_t<Args>, Type>...>, Type>
struct_type(bool align, Args&&... field_types);

std::ostream& operator<<(std::ostream&, const Type::Code&);

std::ostream& operator<<(std::ostream&, const Type&);

/**
 * @ingroup types
 * @brief Creates a boolean type
 *
 * @return Type object
 */
[[nodiscard]] Type bool_();

/**
 * @ingroup types
 * @brief Creates a 8-bit signed integer type
 *
 * @return Type object
 */
[[nodiscard]] Type int8();

/**
 * @ingroup types
 * @brief Creates a 16-bit signed integer type
 *
 * @return Type object
 */
[[nodiscard]] Type int16();

/**
 * @ingroup types
 * @brief Creates a 32-bit signed integer type
 *
 * @return Type object
 */
[[nodiscard]] Type int32();

/**
 * @ingroup types
 * @brief Creates a 64-bit signed integer type
 *
 * @return Type object
 */
[[nodiscard]] Type int64();

/**
 * @ingroup types
 * @brief Creates a 8-bit unsigned integer type
 *
 * @return Type object
 */
[[nodiscard]] Type uint8();

/**
 * @ingroup types
 * @brief Creates a 16-bit unsigned integer type
 *
 * @return Type object
 */
[[nodiscard]] Type uint16();

/**
 * @ingroup types
 * @brief Creates a 32-bit unsigned integer type
 *
 * @return Type object
 */
[[nodiscard]] Type uint32();

/**
 * @ingroup types
 * @brief Creates a 64-bit unsigned integer type
 *
 * @return Type object
 */
[[nodiscard]] Type uint64();

/**
 * @ingroup types
 * @brief Creates a half-precision floating point type
 *
 * @return Type object
 */
[[nodiscard]] Type float16();

/**
 * @ingroup types
 * @brief Creates a single-precision floating point type
 *
 * @return Type object
 */
[[nodiscard]] Type float32();

/**
 * @ingroup types
 * @brief Creates a double-precision floating point type
 *
 * @return Type object
 */
[[nodiscard]] Type float64();

/**
 * @ingroup types
 * @brief Creates a single-precision complex number type
 *
 * @return Type object
 */
[[nodiscard]] Type complex64();

/**
 * @ingroup types
 * @brief Creates a double-precision complex number type
 *
 * @return Type object
 */
[[nodiscard]] Type complex128();

/**
 * @ingroup types
 * @brief Creates a point type
 *
 * @param ndim Number of dimensions
 *
 * @return Type object
 */
[[nodiscard]] Type point_type(uint32_t ndim);

/**
 * @ingroup types
 * @brief Creates a rect type
 *
 * @param ndim Number of dimensions
 *
 * @return Type object
 */
[[nodiscard]] Type rect_type(uint32_t ndim);

/**
 * @ingroup types
 * @brief Creates a null type
 *
 * @return Type object
 */
[[nodiscard]] Type null_type();

/**
 * @ingroup types
 * @brief Checks if the type is a point type of the given dimensionality
 *
 * @param type Type to check
 * @param ndim Number of dimensions the point type should have
 *
 * @return true If the `type` is a point type
 * @return false Otherwise
 */
[[nodiscard]] bool is_point_type(const Type& type, uint32_t ndim);

/**
 * @ingroup types
 * @brief Checks if the type is a rect type of the given dimensionality
 *
 * @param type Type to check
 * @param ndim Number of dimensions the rect type should have
 *
 * @return true If the `type` is a rect type
 * @return false Otherwise
 */
[[nodiscard]] bool is_rect_type(const Type& type, uint32_t ndim);

}  // namespace legate

#include "core/type/type_info.inl"
