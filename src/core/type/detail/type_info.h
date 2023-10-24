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

#include <cstddef>

#include "core/type/type_info.h"

namespace legate::detail {

class BufferBuilder;
class FixedArrayType;
class ListType;
class StructType;

class Type {
 public:
  using Code = legate::Type::Code;

 public:
  Type(Code code);

 public:
  virtual ~Type() {}
  virtual uint32_t size() const;
  virtual uint32_t alignment() const             = 0;
  virtual int32_t uid() const                    = 0;
  virtual bool variable_size() const             = 0;
  virtual std::string to_string() const          = 0;
  virtual bool is_primitive() const              = 0;
  virtual void pack(BufferBuilder& buffer) const = 0;
  virtual const FixedArrayType& as_fixed_array_type() const;
  virtual const StructType& as_struct_type() const;
  virtual const ListType& as_list_type() const;
  virtual bool equal(const Type& other) const = 0;

 public:
  void record_reduction_operator(int32_t op_kind, int64_t global_op_id) const;
  [[nodiscard]] int64_t find_reduction_operator(int32_t op_kind) const;
  [[nodiscard]] int64_t find_reduction_operator(ReductionOpKind op_kind) const;
  bool operator==(const Type& other) const;
  bool operator!=(const Type& other) const { return !operator==(other); }

  const Code code;
};

class PrimitiveType : public Type {
 public:
  PrimitiveType(Code code);
  uint32_t size() const override { return size_; }
  uint32_t alignment() const override { return alignment_; }
  int32_t uid() const override;
  bool variable_size() const override { return false; }
  std::string to_string() const override;
  bool is_primitive() const override { return true; }
  void pack(BufferBuilder& buffer) const override;

 private:
  bool equal(const Type& other) const override;

  uint32_t size_;
  uint32_t alignment_;
};

class StringType : public Type {
 public:
  StringType();
  bool variable_size() const override { return true; }
  uint32_t alignment() const override { return alignof(std::max_align_t); }
  int32_t uid() const override;
  std::string to_string() const override;
  bool is_primitive() const override { return false; }
  void pack(BufferBuilder& buffer) const override;

 private:
  bool equal(const Type& other) const override;
};

class ExtensionType : public Type {
 public:
  ExtensionType(int32_t uid, Type::Code code);
  int32_t uid() const override { return uid_; }
  bool is_primitive() const override { return false; }

 protected:
  const uint32_t uid_;
};

class BinaryType : public ExtensionType {
 public:
  BinaryType(int32_t uid, uint32_t size);
  uint32_t size() const override { return size_; }
  uint32_t alignment() const override { return alignof(std::max_align_t); }
  bool variable_size() const override { return false; }
  std::string to_string() const override;
  void pack(BufferBuilder& buffer) const override;

 private:
  bool equal(const Type& other) const override;

 private:
  const uint32_t size_;
};

class FixedArrayType : public ExtensionType {
 public:
  FixedArrayType(int32_t uid, std::shared_ptr<Type> element_type, uint32_t N);
  uint32_t size() const override { return size_; }
  uint32_t alignment() const override { return element_type_->alignment(); }
  bool variable_size() const override { return false; }
  std::string to_string() const override;
  void pack(BufferBuilder& buffer) const override;
  const FixedArrayType& as_fixed_array_type() const override;

  uint32_t num_elements() const { return N_; }
  std::shared_ptr<Type> element_type() const { return element_type_; }

 private:
  bool equal(const Type& other) const override;

 private:
  const std::shared_ptr<Type> element_type_;
  const uint32_t N_;
  const uint32_t size_;
};

class StructType : public ExtensionType {
 public:
  StructType(int32_t uid, std::vector<std::shared_ptr<Type>>&& field_types, bool align = false);
  uint32_t size() const override { return size_; }
  uint32_t alignment() const override { return alignment_; }
  bool variable_size() const override { return false; }
  std::string to_string() const override;
  void pack(BufferBuilder& buffer) const override;
  const StructType& as_struct_type() const override;

  uint32_t num_fields() const { return field_types_.size(); }
  std::shared_ptr<Type> field_type(uint32_t field_idx) const;
  const std::vector<std::shared_ptr<Type>>& field_types() const { return field_types_; }
  bool aligned() const { return aligned_; }
  const std::vector<uint32_t>& offsets() const { return offsets_; }

 private:
  bool equal(const Type& other) const override;

 private:
  bool aligned_;
  uint32_t size_;
  uint32_t alignment_;
  std::vector<std::shared_ptr<Type>> field_types_{};
  std::vector<uint32_t> offsets_{};
};

class ListType : public ExtensionType {
 public:
  ListType(int32_t uid, std::shared_ptr<Type> element_type);
  uint32_t alignment() const override { return 0; }
  bool variable_size() const override { return true; }
  std::string to_string() const override;
  void pack(BufferBuilder& buffer) const override;
  const ListType& as_list_type() const override;

  std::shared_ptr<Type> element_type() const { return element_type_; }

 private:
  bool equal(const Type& other) const override;

 private:
  const std::shared_ptr<Type> element_type_;
};

std::shared_ptr<Type> primitive_type(Type::Code code);

std::shared_ptr<Type> string_type();

std::shared_ptr<Type> binary_type(uint32_t size);

std::shared_ptr<Type> fixed_array_type(std::shared_ptr<Type> element_type, uint32_t N);

std::shared_ptr<Type> struct_type(const std::vector<std::shared_ptr<Type>>& field_types,
                                  bool align);

std::shared_ptr<Type> struct_type(std::vector<std::shared_ptr<Type>>&& field_types, bool align);

std::shared_ptr<Type> list_type(std::shared_ptr<Type> element_type);

std::shared_ptr<Type> bool_();
std::shared_ptr<Type> int8();
std::shared_ptr<Type> int16();
std::shared_ptr<Type> int32();
std::shared_ptr<Type> int64();
std::shared_ptr<Type> uint8();
std::shared_ptr<Type> uint16();
std::shared_ptr<Type> uint32();
std::shared_ptr<Type> uint64();
std::shared_ptr<Type> float16();
std::shared_ptr<Type> float32();
std::shared_ptr<Type> float64();
std::shared_ptr<Type> complex64();
std::shared_ptr<Type> complex128();
std::shared_ptr<Type> point_type(int32_t ndim);
std::shared_ptr<Type> rect_type(int32_t ndim);
std::shared_ptr<Type> null_type();
bool is_point_type(const std::shared_ptr<Type>& type, int32_t ndim);
bool is_rect_type(const std::shared_ptr<Type>& type, int32_t ndim);

}  // namespace legate::detail
