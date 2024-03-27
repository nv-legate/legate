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

#define LegateConcat_(x, y) x##y
#define LegateConcat(x, y) LegateConcat_(x, y)

// Each suffix defines an additional "enabled" state for LegateDefined(LEGATE_), i.e. if you define
//
// #define LegateDefinedEnabledForm_FOO ignored,
//                                  ^^^~~~~~~~~~~~ note suffix
// Results in
//
// #define LEGATE_HAVE_BAR FOO
// LegateDefined(LEGATE_HAVE_BAR) // now evalues to 1
#define LegateDefinedEnabledForm_1 ignored,
#define LegateDefinedEnabledForm_ ignored,

// arguments are either
// - (0, 1, 0, dummy)
// - (1, 0, dummy)
// this final step cherry-picks the middle
#define LegateDefinedPrivate___(ignored, val, ...) val
// the following 2 steps are needed purely for MSVC since it has a nonconforming preprocessor
// and does not expand __VA_ARGS__ in a single step
#define LegateDefinedPrivate__(args) LegateDefinedPrivate___ args
#define LegateDefinedPrivate_(...) LegateDefinedPrivate__((__VA_ARGS__))
// We do not want parentheses around 'x' since we need it to be expanded as-is to push the 1
// forward an arg space
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define LegateDefinedPrivate(x) LegateDefinedPrivate_(x 1, 0, dummy)
#define LegateDefined(x) LegateDefinedPrivate(LegateConcat_(LegateDefinedEnabledForm_, x))
