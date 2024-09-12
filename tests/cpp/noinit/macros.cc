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

#include "legate/utilities/macros.h"

#include <string_view>

namespace {

#define FOO 1
#define BAR 2

static_assert(std::string_view{LEGATE_STRINGIZE(FOO, BAR)} == "1, 2");
static_assert(std::string_view{LEGATE_STRINGIZE_(FOO, BAR)} == "FOO, BAR");

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

}  // namespace
