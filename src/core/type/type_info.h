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

/**
 * @ingroup types
 * @brief A base class for data type metadata
 */
class Type {
 public:
  /**
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

 public:
  virtual ~Type() {}

  /**
   * @brief Size of the data type in bytes
   *
   * @return Data type size in bytes
   */
  virtual uint32_t size() const = 0;

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
  int32_t uid() const override;
  bool variable_size() const override { return false; }
  std::unique_ptr<Type> clone() const override;
  std::string to_string() const override;

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
  int32_t uid() const override;
  std::unique_ptr<Type> clone() const override;
  std::string to_string() const override;
};

/**
 * @ingroup types
 * @brief A class for all extension types. Each extension type expects a unique ID.
 */
class ExtensionType : public Type {
 public:
  ExtensionType(int32_t uid, Type::Code code);
  int32_t uid() const override { return uid_; }

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
  FixedArrayType(int32_t uid, std::unique_ptr<Type> element_type, uint32_t N);
  uint32_t size() const override { return size_; }
  bool variable_size() const override { return false; }
  std::unique_ptr<Type> clone() const override;
  std::string to_string() const override;
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
  const Type* element_type() const { return element_type_.get(); }

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
   */
  StructType(int32_t uid, std::vector<std::unique_ptr<Type>>&& field_types);
  uint32_t size() const override;
  bool variable_size() const override { return false; }
  std::unique_ptr<Type> clone() const override;
  std::string to_string() const override;
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
  const Type* field_type(uint32_t field_idx) const;

 private:
  std::vector<std::unique_ptr<Type>> field_types_{};
};

/**
 * @ingroup types
 * @brief Creates a metadata object for a primitive type
 *
 * @param uid Unique ID
 * @param element_type Type of the array elements
 * @param N Size of the array
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
 * @param uid Unique ID
 * @param element_type Type of the array elements
 * @param N Size of the array
 *
 * @return Type object
 */
std::unique_ptr<Type> fixed_array_type(int32_t uid, std::unique_ptr<Type> element_type, uint32_t N);

/**
 * @ingroup types
 * @brief Creates a metadata object for a struct type
 *
 * @param uid Unique ID
 * @param field_types A vector of field types
 *
 * @return Type object
 */
std::unique_ptr<Type> struct_type(int32_t uid, std::vector<std::unique_ptr<Type>>&& field_types);

// The caller transfers ownership of the Type objects
std::unique_ptr<Type> struct_type_raw_ptrs(int32_t uid, std::vector<Type*> field_types);

}  // namespace legate
