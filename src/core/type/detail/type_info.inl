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

#include "core/type/detail/type_info.h"

namespace legate::detail {

inline Type::Type(Code code) : code{code} {}

inline bool Type::operator!=(const Type& other) const { return !operator==(other); }

// ==========================================================================================

inline uint32_t PrimitiveType::size() const { return size_; }

inline uint32_t PrimitiveType::alignment() const { return alignment_; }

inline bool PrimitiveType::variable_size() const { return false; }

inline bool PrimitiveType::is_primitive() const { return true; }

// ==========================================================================================

inline bool StringType::variable_size() const { return true; }

inline uint32_t StringType::alignment() const { return alignof(std::max_align_t); }

inline bool StringType::is_primitive() const { return false; }

// ==========================================================================================

inline int32_t ExtensionType::uid() const { return static_cast<std::int32_t>(uid_); }

inline bool ExtensionType::is_primitive() const { return false; }

// ==========================================================================================

inline uint32_t BinaryType::size() const { return size_; }

inline uint32_t BinaryType::alignment() const { return alignof(std::max_align_t); }

inline bool BinaryType::variable_size() const { return false; }

// ==========================================================================================

inline uint32_t FixedArrayType::size() const { return size_; }

inline uint32_t FixedArrayType::alignment() const { return element_type_->alignment(); }

inline bool FixedArrayType::variable_size() const { return false; }

inline uint32_t FixedArrayType::num_elements() const { return N_; }

inline const std::shared_ptr<Type>& FixedArrayType::element_type() const { return element_type_; }

// ==========================================================================================

inline uint32_t StructType::size() const { return size_; }

inline uint32_t StructType::alignment() const { return alignment_; }

inline bool StructType::variable_size() const { return false; }

inline uint32_t StructType::num_fields() const { return field_types().size(); }

inline const std::vector<std::shared_ptr<Type>>& StructType::field_types() const
{
  return field_types_;
}

inline bool StructType::aligned() const { return aligned_; }

inline const std::vector<uint32_t>& StructType::offsets() const { return offsets_; }

// ==========================================================================================

inline uint32_t ListType::alignment() const { return 0; }

inline bool ListType::variable_size() const { return true; }

inline const std::shared_ptr<Type>& ListType::element_type() const { return element_type_; }

}  // namespace legate::detail
