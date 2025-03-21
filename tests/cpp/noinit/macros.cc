/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/macros.h>

#include <string_view>

namespace {

#define FOO 1
#define BAR 2

static_assert(std::string_view{LEGATE_STRINGIZE(FOO, BAR)} == "1, 2");
static_assert(std::string_view{LEGATE_STRINGIZE_(FOO, BAR)} == "FOO, BAR");

// Obviously these are all redundant expressions, that's the point of the
// static_assert(). Thanks for that, clang-tidy!
// NOLINTBEGIN(misc-redundant-expression)
static_assert(LEGATE_CONCAT(FOO, BAR) == 12);  // NOLINT(readability-magic-numbers)

#define FOOBAR 45

static_assert(LEGATE_CONCAT_(FOO, BAR) == FOOBAR);

/// [LEGATE_DEFINED]
#define FOO_EMPTY
#define FOO_ONE 1
#define FOO_ZERO 0
// #define FOO_UNDEFINED

static_assert(LEGATE_DEFINED(FOO_EMPTY) == 1);
static_assert(LEGATE_DEFINED(FOO_ONE) == 1);
static_assert(LEGATE_DEFINED(FOO_ZERO) == 0);
static_assert(LEGATE_DEFINED(FOO_UNDEFINED) == 0);
/// [LEGATE_DEFINED]
// NOLINTEND(misc-redundant-expression)

}  // namespace
