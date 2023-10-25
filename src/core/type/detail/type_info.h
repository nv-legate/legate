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

#include "core/type/type_info.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace legate::detail {

class BufferBuilder;
class FixedArrayType;
class ListType;
class StructType;

class Type {
 public:
  using Code = legate::Type::Code;

  explicit Type(Code code);

  virtual ~Type() = default;
  [[nodiscard]] virtual uint32_t size() const;
  [[nodiscard]] virtual uint32_t alignment() const    = 0;
  [[nodiscard]] virtual int32_t uid() const           = 0;
  [[nodiscard]] virtual bool variable_size() const    = 0;
  [[nodiscard]] virtual std::string to_string() const = 0;
  [[nodiscard]] virtual bool is_primitive() const     = 0;
  virtual void pack(BufferBuilder& buffer) const      = 0;
  [[nodiscard]] virtual const FixedArrayType& as_fixed_array_type() const;
  [[nodiscard]] virtual const StructType& as_struct_type() const;
  [[nodiscard]] virtual const ListType& as_list_type() const;
  [[nodiscard]] virtual bool equal(const Type& other) const = 0;

  void record_reduction_operator(int32_t op_kind, int64_t global_op_id) const;
  [[nodiscard]] int64_t find_reduction_operator(int32_t op_kind) const;
  [[nodiscard]] int64_t find_reduction_operator(ReductionOpKind op_kind) const;
  bool operator==(const Type& other) const;
  bool operator!=(const Type& other) const;

  Code code;
};

class PrimitiveType : public Type {
 public:
  explicit PrimitiveType(Code code);

  [[nodiscard]] uint32_t size() const override;
  [[nodiscard]] uint32_t alignment() const override;
  [[nodiscard]] int32_t uid() const override;
  [[nodiscard]] bool variable_size() const override;
  [[nodiscard]] std::string to_string() const override;
  [[nodiscard]] bool is_primitive() const override;
  void pack(BufferBuilder& buffer) const override;

 private:
  [[nodiscard]] bool equal(const Type& other) const override;

  uint32_t size_{};
  uint32_t alignment_{};
};

class StringType : public Type {
 public:
  StringType();

  [[nodiscard]] bool variable_size() const override;
  [[nodiscard]] uint32_t alignment() const override;
  [[nodiscard]] int32_t uid() const override;
  [[nodiscard]] std::string to_string() const override;
  [[nodiscard]] bool is_primitive() const override;
  void pack(BufferBuilder& buffer) const override;

 private:
  [[nodiscard]] bool equal(const Type& other) const override;
};

class ExtensionType : public Type {
 public:
  ExtensionType(int32_t uid, Type::Code code);
  [[nodiscard]] int32_t uid() const override;
  [[nodiscard]] bool is_primitive() const override;

 protected:
  uint32_t uid_{};
};

class BinaryType : public ExtensionType {
 public:
  BinaryType(int32_t uid, uint32_t size);

  [[nodiscard]] uint32_t size() const override;
  [[nodiscard]] uint32_t alignment() const override;
  [[nodiscard]] bool variable_size() const override;
  [[nodiscard]] std::string to_string() const override;
  void pack(BufferBuilder& buffer) const override;

 private:
  [[nodiscard]] bool equal(const Type& other) const override;

  uint32_t size_{};
};

class FixedArrayType : public ExtensionType {
 public:
  FixedArrayType(int32_t uid, std::shared_ptr<Type> element_type, uint32_t N);

  [[nodiscard]] uint32_t size() const override;
  [[nodiscard]] uint32_t alignment() const override;
  [[nodiscard]] bool variable_size() const override;
  [[nodiscard]] std::string to_string() const override;
  void pack(BufferBuilder& buffer) const override;
  [[nodiscard]] const FixedArrayType& as_fixed_array_type() const override;

  [[nodiscard]] uint32_t num_elements() const;
  [[nodiscard]] const std::shared_ptr<Type>& element_type() const;

 private:
  [[nodiscard]] bool equal(const Type& other) const override;

  std::shared_ptr<Type> element_type_{};
  uint32_t N_{};
  uint32_t size_{};
};

class StructType : public ExtensionType {
 public:
  StructType(int32_t uid, std::vector<std::shared_ptr<Type>>&& field_types, bool align = false);

  [[nodiscard]] uint32_t size() const override;
  [[nodiscard]] uint32_t alignment() const override;
  [[nodiscard]] bool variable_size() const override;
  [[nodiscard]] std::string to_string() const override;
  void pack(BufferBuilder& buffer) const override;
  [[nodiscard]] const StructType& as_struct_type() const override;

  [[nodiscard]] uint32_t num_fields() const;
  [[nodiscard]] std::shared_ptr<Type> field_type(uint32_t field_idx) const;
  [[nodiscard]] const std::vector<std::shared_ptr<Type>>& field_types() const;
  [[nodiscard]] bool aligned() const;
  [[nodiscard]] const std::vector<uint32_t>& offsets() const;

 private:
  [[nodiscard]] bool equal(const Type& other) const override;

  bool aligned_{};
  uint32_t size_{};
  uint32_t alignment_{};
  std::vector<std::shared_ptr<Type>> field_types_{};
  std::vector<uint32_t> offsets_{};
};

class ListType : public ExtensionType {
 public:
  ListType(int32_t uid, std::shared_ptr<Type> element_type);

  [[nodiscard]] uint32_t alignment() const override;
  [[nodiscard]] bool variable_size() const override;
  [[nodiscard]] std::string to_string() const override;
  void pack(BufferBuilder& buffer) const override;
  [[nodiscard]] const ListType& as_list_type() const override;

  [[nodiscard]] const std::shared_ptr<Type>& element_type() const;

 private:
  [[nodiscard]] bool equal(const Type& other) const override;

  std::shared_ptr<Type> element_type_{};
};

[[nodiscard]] std::shared_ptr<Type> primitive_type(Type::Code code);

[[nodiscard]] std::shared_ptr<Type> string_type();

[[nodiscard]] std::shared_ptr<Type> binary_type(uint32_t size);

[[nodiscard]] std::shared_ptr<Type> fixed_array_type(std::shared_ptr<Type> element_type,
                                                     uint32_t N);

[[nodiscard]] std::shared_ptr<Type> struct_type(
  const std::vector<std::shared_ptr<Type>>& field_types, bool align);

[[nodiscard]] std::shared_ptr<Type> struct_type(std::vector<std::shared_ptr<Type>>&& field_types,
                                                bool align);

[[nodiscard]] std::shared_ptr<Type> list_type(std::shared_ptr<Type> element_type);

[[nodiscard]] std::shared_ptr<Type> bool_();
[[nodiscard]] std::shared_ptr<Type> int8();
[[nodiscard]] std::shared_ptr<Type> int16();
[[nodiscard]] std::shared_ptr<Type> int32();
[[nodiscard]] std::shared_ptr<Type> int64();
[[nodiscard]] std::shared_ptr<Type> uint8();
[[nodiscard]] std::shared_ptr<Type> uint16();
[[nodiscard]] std::shared_ptr<Type> uint32();
[[nodiscard]] std::shared_ptr<Type> uint64();
[[nodiscard]] std::shared_ptr<Type> float16();
[[nodiscard]] std::shared_ptr<Type> float32();
[[nodiscard]] std::shared_ptr<Type> float64();
[[nodiscard]] std::shared_ptr<Type> complex64();
[[nodiscard]] std::shared_ptr<Type> complex128();
[[nodiscard]] std::shared_ptr<Type> point_type(int32_t ndim);
[[nodiscard]] std::shared_ptr<Type> rect_type(int32_t ndim);
[[nodiscard]] std::shared_ptr<Type> null_type();
[[nodiscard]] bool is_point_type(const std::shared_ptr<Type>& type, int32_t ndim);
[[nodiscard]] bool is_rect_type(const std::shared_ptr<Type>& type, int32_t ndim);

}  // namespace legate::detail

#include "core/type/detail/type_info.inl"
