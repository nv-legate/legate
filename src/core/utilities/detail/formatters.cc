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

#include "core/utilities/detail/formatters.h"

#include "core/data/detail/shape.h"
#include "core/operation/detail/operation.h"
#include "core/partitioning/detail/constraint.h"
#include "core/type/detail/type_info.h"
#include "core/utilities/typedefs.h"

#include <fmt/format.h>

namespace fmt {

format_context::iterator formatter<legate::detail::Type>::format(const legate::detail::Type& a,
                                                                 format_context& ctx) const
{
  return formatter<std::string>::format(a.to_string(), ctx);
}

format_context::iterator formatter<legate::detail::Operation>::format(
  const legate::detail::Operation& op, format_context& ctx) const
{
  return formatter<std::string>::format(op.to_string(), ctx);
}

format_context::iterator formatter<legate::detail::Shape>::format(
  const legate::detail::Shape& shape, format_context& ctx) const
{
  return formatter<std::string>::format(shape.to_string(), ctx);
}

format_context::iterator formatter<legate::detail::Constraint>::format(
  const legate::detail::Constraint& constraint, format_context& ctx) const
{
  return formatter<std::string>::format(constraint.to_string(), ctx);
}

format_context::iterator formatter<legate::detail::Expr>::format(const legate::detail::Expr& expr,
                                                                 format_context& ctx) const
{
  return formatter<std::string>::format(expr.to_string(), ctx);
}

format_context::iterator formatter<legate::VariantCode>::format(legate::VariantCode variant,
                                                                format_context& ctx) const
{
  string_view name = "(unknown)";

  switch (variant) {
#define LEGATE_VARIANT_CASE(x) \
  case legate::VariantCode::x: name = #x "_VARIANT"; break
    LEGATE_VARIANT_CASE(NONE);
    LEGATE_VARIANT_CASE(CPU);
    LEGATE_VARIANT_CASE(GPU);
    LEGATE_VARIANT_CASE(OMP);
#undef LEGATE_VARIANT_CASE
  }

  return formatter<string_view>::format(name, ctx);
}

format_context::iterator formatter<legate::LocalTaskID>::format(legate::LocalTaskID id,
                                                                format_context& ctx) const
{
  return formatter<std::underlying_type_t<legate::LocalTaskID>>::format(fmt::underlying(id), ctx);
}

format_context::iterator formatter<legate::GlobalTaskID>::format(legate::GlobalTaskID id,
                                                                 format_context& ctx) const
{
  return formatter<std::underlying_type_t<legate::GlobalTaskID>>::format(fmt::underlying(id), ctx);
}

}  // namespace fmt
