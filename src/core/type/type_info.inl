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

namespace legate {

inline Type::Type(InternalSharedPtr<detail::Type> impl) : impl_{std::move(impl)} {}

inline const SharedPtr<detail::Type>& Type::impl() const { return impl_; }

// ==========================================================================================

inline FixedArrayType::FixedArrayType(InternalSharedPtr<detail::Type> type) : Type{std::move(type)}
{
}

// ==========================================================================================

inline StructType::StructType(InternalSharedPtr<detail::Type> type) : Type{std::move(type)} {}

// ==========================================================================================

inline ListType::ListType(InternalSharedPtr<detail::Type> type) : Type{std::move(type)} {}

// ==========================================================================================

template <typename... Args>
std::enable_if_t<std::conjunction_v<std::is_same<std::decay_t<Args>, Type>...>, Type> struct_type(
  bool align, Args&&... field_types)
{
  std::vector<Type> vec_field_types;

  vec_field_types.reserve(sizeof...(field_types));
  (vec_field_types.emplace_back(std::forward<Args>(field_types)), ...);
  return struct_type(vec_field_types, align);
}

}  // namespace legate
