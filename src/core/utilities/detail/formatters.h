/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "core/legate_c.h"

#include <fmt/format.h>
#include <string>

namespace legate::detail {

class Type;
class Operation;
class Shape;
class Constraint;
class Expr;

}  // namespace legate::detail

namespace fmt {

template <>
struct formatter<legate::detail::Type> : formatter<std::string> {
  format_context::iterator format(const legate::detail::Type& a, format_context& ctx) const;
};

template <>
struct formatter<legate::detail::Operation> : formatter<std::string> {
  format_context::iterator format(const legate::detail::Operation& op, format_context& ctx) const;
};

template <>
struct formatter<legate::detail::Shape> : formatter<std::string> {
  format_context::iterator format(const legate::detail::Shape& shape, format_context& ctx) const;
};

template <>
struct formatter<legate::detail::Constraint> : formatter<std::string> {
  format_context::iterator format(const legate::detail::Constraint& constraint,
                                  format_context& ctx) const;
};

template <typename T, typename Char>
struct formatter<T, Char, std::enable_if_t<std::is_base_of_v<legate::detail::Constraint, T>>>
  : formatter<legate::detail::Constraint, Char> {
  format_context::iterator format(const T& constraint, format_context& ctx) const
  {
    return formatter<legate::detail::Constraint, Char>::format(constraint, ctx);
  }
};

template <>
struct formatter<legate::detail::Expr> : formatter<std::string> {
  format_context::iterator format(const legate::detail::Expr& expr, format_context& ctx) const;
};

template <typename T, typename Char>
struct formatter<T, Char, std::enable_if_t<std::is_base_of_v<legate::detail::Expr, T>>>
  : formatter<legate::detail::Expr, Char> {
  format_context::iterator format(const T& expr, format_context& ctx) const
  {
    return formatter<legate::detail::Expr, Char>::format(expr, ctx);
  }
};

template <>
struct formatter<legate_core_variant_t> : formatter<string_view> {
  format_context::iterator format(legate_core_variant_t variant, format_context& ctx) const;
};

}  // namespace fmt
