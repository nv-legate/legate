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

#pragma once

#include "core/legate_c.h"

#include <memory>
#include <vector>

/** @defgroup types Type system
 */

namespace legate {

class BufferBuilder;
class FixedArrayType;
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
    FIXED_ARRAY = FIXED_ARRAY_LT, /*!< Fixed-size array type */
    STRUCT      = STRUCT_LT,      /*!< Struct type */
    STRING      = STRING_LT,      /*!< String type */
    INVALID     = INVALID_LT,     /*!< Invalid type */
  };

 protected:
  Type(Code code);

  virtual bool equal(const Type& other) const = 0;

 public:
  virtual ~Type() {}

  /**
   * @brief Size of the data type in bytes
   *
   * @return Data type size in bytes
   */
  virtual uint32_t size() const = 0;

  /**
   * @brief Alignment of the type
   *
   * @return Alignment in bytes
   */
  virtual uint32_t alignment() const = 0;

  /**
   * @brief Unique ID of the data type
   *
   * @return Unique ID
   */
  virtual int32_t uid() const = 0;

  /**
   * @brief Inidicates whether the data type is of varible size elements
   *
   * @return true Elements can be variable size
   * @return false Elements have fixed size
   */
  virtual bool variable_size() const = 0;

  /**
   * @brief Returns a copy of the data type
   *
   * @return A copy of the data type
   */
  virtual std::unique_ptr<Type> clone() const = 0;

  /**
   * @brief Converts the data type into a string
   *
   * @return A string of the data type
   */
  virtual std::string to_string() const = 0;

  /**
   * @brief Indicates whether the type is a primitive type
   *
   * @return true If the type is a primitive type
   * @return false Otherwise
   */
  virtual bool is_primitive() const = 0;
  /**
   * @brief Serializes the type into a buffer
   *
   * @param buffer A BufferBuilder object to serialize the type into
   */
  virtual void pack(BufferBuilder& buffer) const = 0;

  /**
   * @brief Dynamically casts the type into a fixed size array type.
   *
   * If the type is not a fixed size array type, an exception will be raised.
   *
   * @return Type object
   */
  virtual const FixedArrayType& as_fixed_array_type() const;

  /**
   * @brief Dynamically casts the type into a struct type.
   *
   * If the type is not a struct type, an exception will be raised.
   *
   * @return Type object
   */
  virtual const StructType& as_struct_type() const;

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
   * @brief Finds the global operator ID for a given reduction operator kind.
   *
   * Raises an exception if no reduction operator has been registered for the kind.
   *
   * @param op_kind Reduction operator kind
   *
   * @return Global reduction operator ID
   */
  int32_t find_reduction_operator(int32_t op_kind) const;

  /**
   * @brief Finds the global operator ID for a given reduction operator kind.
   *
   * Raises an exception if no reduction operator has been registered for the kind.
   *
   * @param op_kind Reduction operator kind
   *
   * @return Global reduction operator ID
   */
  int32_t find_reduction_operator(ReductionOpKind op_kind) const;

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
  bool operator!=(const Type& other) const { return !operator==(other); }

  const Code code;
};

/**
 * @ingroup types
 * @brief A class for primitive data types
 */
class PrimitiveType : public Type {
 public:
  /**
   * @brief Constructs a primitive type metadata object
   *
   * @param code Type code. Must be one of the primitive types.
   */
  PrimitiveType(Code code);
  uint32_t size() const override { return size_; }
  uint32_t alignment() const override { return size_; }
  int32_t uid() const override;
  bool variable_size() const override { return false; }
  std::unique_ptr<Type> clone() const override;
  std::string to_string() const override;
  bool is_primitive() const override { return true; }
  void pack(BufferBuilder& buffer) const override;

 private:
  bool equal(const Type& other) const override;

 private:
  const uint32_t size_;
};

/**
 * @ingroup types
 * @brief String data type
 */
class StringType : public Type {
 public:
  StringType();
  bool variable_size() const override { return true; }
  uint32_t size() const override { return 0; }
  uint32_t alignment() const override { return 0; }
  int32_t uid() const override;
  std::unique_ptr<Type> clone() const override;
  std::string to_string() const override;
  bool is_primitive() const override { return false; }
  void pack(BufferBuilder& buffer) const override;

 private:
  bool equal(const Type& other) const override;
};

/**
 * @ingroup types
 * @brief A class for all extension types. Each extension type expects a unique ID.
 */
class ExtensionType : public Type {
 public:
  ExtensionType(int32_t uid, Type::Code code);
  int32_t uid() const override { return uid_; }
  bool is_primitive() const override { return false; }

 protected:
  const uint32_t uid_;
};

/**
 * @ingroup types
 * @brief A class for fixed-size array data types
 */
class FixedArrayType : public ExtensionType {
 public:
  /**
   * @brief Constructs a metadata object for a fixed-size array type
   *
   * @param uid Unique ID
   * @param element_type Type of the array elements
   * @param N Size of the array
   */
  FixedArrayType(int32_t uid, std::unique_ptr<Type> element_type, uint32_t N) noexcept(false);
  uint32_t size() const override { return size_; }
  uint32_t alignment() const override { return element_type_->alignment(); }
  bool variable_size() const override { return false; }
  std::unique_ptr<Type> clone() const override;
  std::string to_string() const override;
  void pack(BufferBuilder& buffer) const override;
  const FixedArrayType& as_fixed_array_type() const override;

  /**
   * @brief Returns the number of elements
   *
   * @return Number of elements
   */
  uint32_t num_elements() const { return N_; }
  /**
   * @brief Returns the element type
   *
   * @return Element type
   */
  const Type& element_type() const { return *element_type_; }

 private:
  bool equal(const Type& other) const override;

 private:
  const std::unique_ptr<Type> element_type_;
  const uint32_t N_;
  const uint32_t size_;
};

/**
 * @ingroup types
 * @brief A class for struct data types
 */
class StructType : public ExtensionType {
 public:
  /**
   * @brief Constructs a metadata object for a struct type
   *
   * @param uid Unique ID
   * @param field_types A vector of field types
   * @param align Optional boolean flag indicating whether the struct fields should be aligned.
   *              false by default.
   */
  StructType(int32_t uid,
             std::vector<std::unique_ptr<Type>>&& field_types,
             bool align = false) noexcept(false);
  uint32_t size() const override { return size_; }
  uint32_t alignment() const override { return alignment_; }
  bool variable_size() const override { return false; }
  std::unique_ptr<Type> clone() const override;
  std::string to_string() const override;
  void pack(BufferBuilder& buffer) const override;
  const StructType& as_struct_type() const override;

  /**
   * @brief Returns the number of fields
   *
   * @return Number of fields
   */
  uint32_t num_fields() const { return field_types_.size(); }
  /**
   * @brief Returns the element type
   *
   * @param field_idx Field index. Must be within the range
   *
   * @return Element type
   */
  const Type& field_type(uint32_t field_idx) const;
  /**
   * @brief Indiciates whether the fields are aligned
   *
   * @return true Fields are aligned
   * @return false Fields are compact
   */
  bool aligned() const { return aligned_; }

 private:
  bool equal(const Type& other) const override;

 private:
  bool aligned_;
  uint32_t size_;
  uint32_t alignment_;
  std::vector<std::unique_ptr<Type>> field_types_{};
  std::vector<uint32_t> offsets_{};
};

/**
 * @ingroup types
 * @brief Creates a metadata object for a primitive type
 *
 * @param code Type code
 *
 * @return Type object
 */
std::unique_ptr<Type> primitive_type(Type::Code code);

/**
 * @ingroup types
 * @brief Creates a metadata object for the string type
 *
 * @return Type object
 */
std::unique_ptr<Type> string_type();

/**
 * @ingroup types
 * @brief Creates a metadata object for a fixed-size array type
 *
 * @param element_type Type of the array elements
 * @param N Size of the array
 *
 * @return Type object
 */
std::unique_ptr<Type> fixed_array_type(std::unique_ptr<Type> element_type,
                                       uint32_t N) noexcept(false);

/**
 * @ingroup types
 * @brief Creates a metadata object for a struct type
 *
 * @param field_types A vector of field types
 * @param align If true, fields in the struct are aligned
 *
 * @return Type object
 */
std::unique_ptr<Type> struct_type(std::vector<std::unique_ptr<Type>>&& field_types,
                                  bool align = false) noexcept(false);

/**
 * @ingroup types
 * @brief Creates a metadata object for a struct type
 *
 * @param align If true, fields in the struct are aligned
 * @param field_types Field types
 *
 * @return Type object
 */
template <class... Types>
std::enable_if_t<std::conjunction_v<std::is_same<Types, Type>...>, std::unique_ptr<Type>>
struct_type(bool align, std::unique_ptr<Types>... field_types) noexcept(false)
{
  std::vector<std::unique_ptr<Type>> vec_field_types;
  auto move_field_type = [&vec_field_types](auto&& field_type) {
    vec_field_types.push_back(std::move(field_type));
  };
  (move_field_type(std::move(field_types)), ...);
  return struct_type(std::move(vec_field_types), align);
}

std::ostream& operator<<(std::ostream&, const Type::Code&);

std::ostream& operator<<(std::ostream&, const Type&);

/**
 * @ingroup types
 * @brief Creates a boolean type
 *
 * @return Type object
 */
std::unique_ptr<Type> bool_();

/**
 * @ingroup types
 * @brief Creates a 8-bit signed integer type
 *
 * @return Type object
 */
std::unique_ptr<Type> int8();

/**
 * @ingroup types
 * @brief Creates a 16-bit signed integer type
 *
 * @return Type object
 */
std::unique_ptr<Type> int16();

/**
 * @ingroup types
 * @brief Creates a 32-bit signed integer type
 *
 * @return Type object
 */
std::unique_ptr<Type> int32();

/**
 * @ingroup types
 * @brief Creates a 64-bit signed integer type
 *
 * @return Type object
 */
std::unique_ptr<Type> int64();

/**
 * @ingroup types
 * @brief Creates a 8-bit unsigned integer type
 *
 * @return Type object
 */
std::unique_ptr<Type> uint8();

/**
 * @ingroup types
 * @brief Creates a 16-bit unsigned integer type
 *
 * @return Type object
 */
std::unique_ptr<Type> uint16();

/**
 * @ingroup types
 * @brief Creates a 32-bit unsigned integer type
 *
 * @return Type object
 */
std::unique_ptr<Type> uint32();

/**
 * @ingroup types
 * @brief Creates a 64-bit unsigned integer type
 *
 * @return Type object
 */
std::unique_ptr<Type> uint64();

/**
 * @ingroup types
 * @brief Creates a half-precision floating point type
 *
 * @return Type object
 */
std::unique_ptr<Type> float16();

/**
 * @ingroup types
 * @brief Creates a single-precision floating point type
 *
 * @return Type object
 */
std::unique_ptr<Type> float32();

/**
 * @ingroup types
 * @brief Creates a double-precision floating point type
 *
 * @return Type object
 */
std::unique_ptr<Type> float64();

/**
 * @ingroup types
 * @brief Creates a single-precision complex number type
 *
 * @return Type object
 */
std::unique_ptr<Type> complex64();

/**
 * @ingroup types
 * @brief Creates a double-precision complex number type
 *
 * @return Type object
 */
std::unique_ptr<Type> complex128();

/**
 * @ingroup types
 * @brief Creates a string type
 *
 * @return Type object
 */
std::unique_ptr<Type> string();

/**
 * @ingroup types
 * @brief Creates a point type
 *
 * @param ndim Number of dimensions
 *
 * @return Type object
 */
std::unique_ptr<Type> point_type(int32_t ndim);

/**
 * @ingroup types
 * @brief Creates a rect type
 *
 * @param ndim Number of dimensions
 *
 * @return Type object
 */
std::unique_ptr<Type> rect_type(int32_t ndim);

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
bool is_point_type(const Type& type, int32_t ndim);

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
bool is_rect_type(const Type& type, int32_t ndim);

}  // namespace legate
