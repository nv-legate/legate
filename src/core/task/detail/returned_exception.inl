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

#include "core/task/detail/returned_exception.h"

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
  return std::visit(std::forward<T>(fn), variant_);
}

// ==========================================================================================

}  // namespace legate::detail
