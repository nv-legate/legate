/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/returned_exception.h>

namespace legate::detail {

template <typename T, typename SFINAE>
ReturnedException::ReturnedException(T&& t) noexcept(
  std::is_nothrow_constructible_v<variant_type, T>)
  : variant_{std::forward<T>(t)}
{
}

template <typename T, typename U, typename... V>
ReturnedException::ReturnedException(T&& arg1, U&& arg2, V&&... rest) noexcept(
  std::is_nothrow_constructible_v<variant_type, T, U, V...>)
  : variant_{std::forward<T>(arg1), std::forward<U>(arg2), std::forward<V>(rest)...}
{
}

template <typename T>
decltype(auto) ReturnedException::visit(T&& fn)
{
  return std::visit(std::forward<T>(fn), variant_);
}

template <typename T>
decltype(auto) ReturnedException::visit(T&& fn) const
{
  return std::visit(std::forward<T>(fn), variant());
}

inline const ReturnedException::variant_type& ReturnedException::variant() const
{
  return variant_;
}

}  // namespace legate::detail
