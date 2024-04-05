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

/** @addtogroup util
 *  @{
 */

/**
 * @file
 * @brief Definitions of preprocessor utilities.
 */

/**
 * @def LegateConcat_(x, ...)
 *
 * @brief Concatenate a series of tokens without macro expansion.
 *
 * @param x The first parameter to concatenate.
 * @param ... The remaining parameters to concatenate.
 *
 * This macro will NOT macro-expand any tokens passed to it. If this behavior is undesirable,
 * and the user wishes to have all tokens expanded before concatenation, use LegateConcat()
 * instead. For example:
 *
 * @code
 * #define FOO 1
 * #define BAR 2
 *
 * LegateConcat(FOO, BAR) // expands to FOOBAR
 * @endcode
 *
 * @see LegateConcat()
 */
#define LegateConcat_(x, ...) x##__VA_ARGS__

/**
 * @def LegateConcat(x, ...)
 *
 * @brief Concatenate a series of tokens.
 *
 * @param x The first parameter to concatenate.
 * @param ... The remaining parameters to concatenate.
 *
 * This macro will first macro-expand any tokens passed to it. If this behavior is undesirable,
 * use LegateConcat_() instead. For example:
 *
 * @code
 * #define FOO 1
 * #define BAR 2
 *
 * LegateConcat(FOO, BAR) // expands to 12
 * @endcode
 *
 * @see LegateConcat_()
 */
#define LegateConcat(x, ...) LegateConcat_(x, __VA_ARGS__)

/**
 * @def LegateStringize_(...)
 *
 * @brief Stringize a series of tokens.
 *
 * @param ... The tokens to stringize.
 *
 * This macro will turn its arguments into compile-time constant C strings.
 *
 * This macro will NOT macro-expand any tokens passed to it. If this behavior is undesirable,
 * and the user wishes to have all tokens expanded before stringification, use
 * LegateStringize() instead. For example:
 *
 * @code
 * #define FOO 1
 * #define BAR 2
 *
 * LegateStringize_(FOO, BAR) // expands to "FOO, BAR" (note the "")
 * @endcode
 *
 * @see LegateStringize()
 */
#define LegateStringize_(...) #__VA_ARGS__

/**
 * @def LegateStringize(...)
 *
 * @brief Stringize a series of tokens.
 *
 * @param ... The tokens to stringize.
 *
 * This macro will turn its arguments into compile-time constant C strings.
 *
 * This macro will first macro-expand any tokens passed to it. If this behavior is undesirable,
 * use LegateStringize_() instead. For example:
 *
 * @code
 * #define FOO 1
 * #define BAR 2
 *
 * LegateStringize(FOO, BAR) // expands to "1, 2" (note the "")
 * @endcode
 *
 * @see LegateStringize_()
 */
#define LegateStringize(...) LegateStringize_(__VA_ARGS__)

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

// NOLINTBEGIN(bugprone-reserved-identifier)
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

/**
 * @def LegateDefined(x)
 *
 * @brief Determine if a preprocessor definition is positively defined.
 *
 * @param x The legate preprocessor definition.
 * @return 1 if the argument is defined and true, 0 otherwise.
 *
 * LegateDefined() returns 1 if and only if \a x expands to integer literal 1, or is defined
 * (but empty). In all other cases, LegateDefined() returns the integer literal 0. Therefore
 * this macro should not be used if its argument may expand to a non-empty value other than 1. The
 * only exception is if the argument is defined but expands to 0, in which case `LegateDefined()`
 * will also expand to 0:
 *
 * @snippet unit/macros.cc LegateDefined
 *
 * Conceptually, `LegateDefined()` is equivalent to
 *
 * @code
 * #if defined(x) && (x == 1 || x == *empty*)
 * // "return" 1
 * #else
 * // "return" 0
 * #endif
 * @endcode
 *
 * As a result this macro works both in preprocessor statements:
 *
 * @code
 * #if LegateDefined(FOO_BAR)
 *   foo_bar_is_defined();
 * #else
 *   foo_bar_is_not_defined();
 * #endif
 * @endcode
 *
 * And in regular C++ code:
 *
 * @code
 * if (LegateDefined(FOO_BAR)) {
 *   foo_bar_is_defined();
 * } else {
 *   foo_bar_is_not_defined();
 * }
 * @endcode
 *
 * Note that in the C++ example above both arms of the if statement must compile. If this is
 * not desired, then -- since `LegateDefined()` produces a compile-time constant expression --
 * the user may use C++17's `if constexpr` to block out one of the arms:
 *
 * @code
 * if constexpr (LegateDefined(FOO_BAR)) {
 *   foo_bar_is_defined();
 * } else {
 *   foo_bar_is_not_defined();
 * }
 * @endcode
 *
 * @see LegateConcat()
 */
#define LegateDefined(x) LegateDefinedPrivate(LegateConcat_(LegateDefinedEnabledForm_, x))
// NOLINTEND(bugprone-reserved-identifier)

/** @} */  // end of group util
