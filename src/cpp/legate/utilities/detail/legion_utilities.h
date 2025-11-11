/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legion/api/types.h>

#include <type_traits>

namespace legate::detail {

/**
 * @brief In-place bitwise OR for Legion::PrivilegeMode.
 *
 * @param lhs The left operand (updated in-place).
 * @param rhs The right operand.
 *
 * @return A copy of `lhs` after being updated.
 */
constexpr Legion::PrivilegeMode operator|=(Legion::PrivilegeMode& lhs, Legion::PrivilegeMode rhs);

/**
 * @brief Bitwise OR for Legion::PrivilegeMode.
 *
 * @param lhs The left operand.
 * @param rhs The right operand.
 *
 * @return Bitwise OR of `lhs` and `rhs`.
 */
[[nodiscard]] constexpr Legion::PrivilegeMode operator|(Legion::PrivilegeMode lhs,
                                                        Legion::PrivilegeMode rhs);

/**
 * @param priv The privilege to check.
 * @param target The target privilege to check for.
 *
 * @return `true` if `priv` has privilege `target`, `false` otherwise.
 */
[[nodiscard]] constexpr bool has_privilege(Legion::PrivilegeMode priv,
                                           Legion::PrivilegeMode target);

/**
 * @brief Return a new privilege with a particular privilege masked out.
 *
 * @param priv The original privilege.
 * @param to_ignore The privilege to mask out.
 *
 * @return The masked privilege.
 */
[[nodiscard]] constexpr Legion::PrivilegeMode ignore_privilege(
  Legion::PrivilegeMode priv, std::underlying_type_t<Legion::PrivilegeMode> to_ignore);

}  // namespace legate::detail

#include <legate/utilities/detail/legion_utilities.inl>
