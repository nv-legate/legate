/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/type/detail/types.h>

namespace legate::detail {

inline Type::Type(Code type_code) : code{type_code} {}

inline bool Type::operator==(const Type& other) const { return code == other.code; }

inline bool Type::operator!=(const Type& other) const { return !operator==(other); }

// ==========================================================================================

inline std::uint32_t PrimitiveType::size() const { return size_; }

inline std::uint32_t PrimitiveType::alignment() const { return alignment_; }

inline std::uint32_t PrimitiveType::uid() const { return static_cast<std::uint32_t>(code); }

inline bool PrimitiveType::variable_size() const { return false; }

inline bool PrimitiveType::is_primitive() const { return true; }

// ==========================================================================================

inline StringType::StringType() : Type{Type::Code::STRING} {}

inline bool StringType::variable_size() const { return true; }

inline std::uint32_t StringType::alignment() const { return alignof(std::max_align_t); }

inline std::uint32_t StringType::uid() const { return static_cast<std::uint32_t>(code); }

inline std::string StringType::to_string() const { return "string"; }

inline bool StringType::is_primitive() const { return false; }

// ==========================================================================================

inline ExtensionType::ExtensionType(std::uint32_t uid, Type::Code type_code)
  : Type{type_code}, uid_{uid}
{
}

inline std::uint32_t ExtensionType::uid() const { return uid_; }

inline bool ExtensionType::is_primitive() const { return false; }

// ==========================================================================================

inline BinaryType::BinaryType(std::uint32_t uid, std::uint32_t size)
  : ExtensionType{uid, Type::Code::BINARY}, size_{size}
{
}

inline std::uint32_t BinaryType::size() const { return size_; }

inline std::uint32_t BinaryType::alignment() const { return alignof(std::max_align_t); }

inline bool BinaryType::variable_size() const { return false; }

inline bool BinaryType::operator==(const Type& other) const { return uid() == other.uid(); }

// ==========================================================================================

inline std::uint32_t FixedArrayType::size() const { return size_; }

inline std::uint32_t FixedArrayType::alignment() const { return element_type()->alignment(); }

inline bool FixedArrayType::variable_size() const { return false; }

inline std::uint32_t FixedArrayType::num_elements() const { return N_; }

inline const InternalSharedPtr<Type>& FixedArrayType::element_type() const { return element_type_; }

// ==========================================================================================

inline std::uint32_t StructType::size() const { return size_; }

inline std::uint32_t StructType::alignment() const { return alignment_; }

inline bool StructType::variable_size() const { return false; }

inline std::uint32_t StructType::num_fields() const
{
  return static_cast<std::uint32_t>(field_types().size());
}

inline const std::vector<InternalSharedPtr<Type>>& StructType::field_types() const
{
  return field_types_;
}

inline bool StructType::aligned() const { return aligned_; }

inline const std::vector<std::uint32_t>& StructType::offsets() const { return offsets_; }

// ==========================================================================================

inline std::uint32_t ListType::alignment() const { return 0; }

inline bool ListType::variable_size() const { return true; }

inline const InternalSharedPtr<Type>& ListType::element_type() const { return element_type_; }

}  // namespace legate::detail
