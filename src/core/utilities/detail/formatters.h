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

#include <fmt/format.h>
#include <string>
#include <type_traits>

namespace legate {

enum class LocalTaskID : std::int64_t;
enum class GlobalTaskID : unsigned int /* A.K.A. Legion::TaskID */;
enum class VariantCode : unsigned int /* A.K.A. Legion::VariantID */;

}  // namespace legate

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
struct formatter<legate::VariantCode> : formatter<string_view> {
  format_context::iterator format(legate::VariantCode variant, format_context& ctx) const;
};

template <>
struct formatter<legate::LocalTaskID> : formatter<std::underlying_type_t<legate::LocalTaskID>> {
  format_context::iterator format(legate::LocalTaskID id, format_context& ctx) const;
};

template <>
struct formatter<legate::GlobalTaskID> : formatter<std::underlying_type_t<legate::GlobalTaskID>> {
  format_context::iterator format(legate::GlobalTaskID id, format_context& ctx) const;
};

}  // namespace fmt
