/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/scalar.h>
#include <legate/data/scalar.h>
#include <legate/type/detail/types.h>
#include <legate/type/type_traits.h>

#include <utility>

namespace legate::detail {

template <typename T>
inline Scalar::Scalar(T value)
  : own_{true},
    type_{detail::primitive_type(type_code_of_v<T>)},
    data_{copy_data_(std::addressof(value), sizeof(T))}
{
}

inline Scalar::Scalar(Scalar&& other) noexcept
  : own_{std::exchange(other.own_, false)},
    type_{std::move(other.type_)},
    data_{std::exchange(other.data_, nullptr)}
{
}

inline const InternalSharedPtr<Type>& Scalar::type() const { return type_; }

inline const void* Scalar::data() const { return data_; }

}  // namespace legate::detail
