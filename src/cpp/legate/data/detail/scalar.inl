/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/scalar.h>
#include <legate/data/scalar.h>
#include <legate/type/type_traits.h>

#include <utility>

namespace legate::detail {

inline Scalar::Scalar(InternalSharedPtr<Type> type)
  : own_{true}, type_{std::move(type)}, data_{new char[type_->size()]{}}
{
}

template <typename T>
inline Scalar::Scalar(T value)
  : own_{true},
    type_{detail::primitive_type(type_code_of_v<T>)},
    data_{copy_data_(std::addressof(value), sizeof(T))}
{
}

inline Scalar::Scalar(const Scalar& other)
  : own_{other.own_},
    type_{other.type_},
    data_{other.own_ ? copy_data_(other.data_, other.size()) : other.data_}
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
