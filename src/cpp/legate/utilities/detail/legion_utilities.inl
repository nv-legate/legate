/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/legion_utilities.h>
#include <legate/utilities/detail/type_traits.h>

namespace legate::detail {

constexpr Legion::PrivilegeMode operator|=(Legion::PrivilegeMode& lhs, Legion::PrivilegeMode rhs)
{
  lhs = lhs | rhs;
  return lhs;
}

constexpr Legion::PrivilegeMode operator|(Legion::PrivilegeMode lhs, Legion::PrivilegeMode rhs)
{
  return Legion::PrivilegeMode{
    to_underlying(lhs) |  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
    to_underlying(rhs)};  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
}

constexpr bool has_privilege(Legion::PrivilegeMode priv, Legion::PrivilegeMode target)
{
  return (priv & target) == target;
}

constexpr Legion::PrivilegeMode ignore_privilege(
  Legion::PrivilegeMode priv, std::underlying_type_t<Legion::PrivilegeMode> to_ignore)
{
  return Legion::PrivilegeMode{priv & (~to_ignore)};
}

}  // namespace legate::detail
