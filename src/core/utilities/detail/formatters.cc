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

format_context::iterator formatter<legate_core_variant_t>::format(legate_core_variant_t variant,
                                                                  format_context& ctx) const
{
  string_view name = "(unknown)";

  switch (variant) {
#define LEGATE_VARIANT_CASE(x) \
  case x: name = #x; break
    LEGATE_VARIANT_CASE(LEGATE_NO_VARIANT);
    LEGATE_VARIANT_CASE(LEGATE_CPU_VARIANT);
    LEGATE_VARIANT_CASE(LEGATE_GPU_VARIANT);
    LEGATE_VARIANT_CASE(LEGATE_OMP_VARIANT);
#undef LEGATE_VARIANT_CASE
  }

  return formatter<string_view>::format(name, ctx);
}

}  // namespace fmt
