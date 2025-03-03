/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// This header must be included here (even though it is not used directly) to ensure that all
// possible uses of LEGATE_DEFINED() use proper legate definitions.
#include <legate_defines.h>  // IWYU pragma: keep

#include <legate/utilities/detail/doxygen.h>

/**
 * @addtogroup util
 * @{
 */

/**
 * @file
 * @brief Definitions of preprocessor utilities.
 */

/**
 * @def LEGATE_CONCAT_(x, ...)
 *
 * @brief Concatenate a series of tokens without macro expansion.
 *
 * @param x The first parameter to concatenate.
 * @param ... The remaining parameters to concatenate.
 *
 * This macro will NOT macro-expand any tokens passed to it. If this behavior is undesirable,
 * and the user wishes to have all tokens expanded before concatenation, use LEGATE_CONCAT()
 * instead. For example:
 *
 * @code
 * #define FOO 1
 * #define BAR 2
 *
 * LEGATE_CONCAT(FOO, BAR) // expands to FOOBAR
 * @endcode
 *
 * @see LEGATE_CONCAT()
 */
#define LEGATE_CONCAT_(x, ...) x##__VA_ARGS__

/**
 * @def LEGATE_CONCAT(x, ...)
 *
 * @brief Concatenate a series of tokens.
 *
 * @param x The first parameter to concatenate.
 * @param ... The remaining parameters to concatenate.
 *
 * This macro will first macro-expand any tokens passed to it. If this behavior is undesirable,
 * use LEGATE_CONCAT_() instead. For example:
 *
 * @code
 * #define FOO 1
 * #define BAR 2
 *
 * LEGATE_CONCAT(FOO, BAR) // expands to 12
 * @endcode
 *
 * @see LEGATE_CONCAT_()
 */
#define LEGATE_CONCAT(x, ...) LEGATE_CONCAT_(x, __VA_ARGS__)

/**
 * @def LEGATE_STRINGIZE_(...)
 *
 * @brief Stringize a series of tokens.
 *
 * @param ... The tokens to stringize.
 *
 * This macro will turn its arguments into compile-time constant C strings.
 *
 * This macro will NOT macro-expand any tokens passed to it. If this behavior is undesirable,
 * and the user wishes to have all tokens expanded before stringification, use
 * LEGATE_STRINGIZE() instead. For example:
 *
 * @code
 * #define FOO 1
 * #define BAR 2
 *
 * LEGATE_STRINGIZE_(FOO, BAR) // expands to "FOO, BAR" (note the "")
 * @endcode
 *
 * @see LEGATE_STRINGIZE()
 */
#define LEGATE_STRINGIZE_(...) #__VA_ARGS__

/**
 * @def LEGATE_STRINGIZE(...)
 *
 * @brief Stringize a series of tokens.
 *
 * @param ... The tokens to stringize.
 *
 * This macro will turn its arguments into compile-time constant C strings.
 *
 * This macro will first macro-expand any tokens passed to it. If this behavior is undesirable,
 * use LEGATE_STRINGIZE_() instead. For example:
 *
 * @code
 * #define FOO 1
 * #define BAR 2
 *
 * LEGATE_STRINGIZE(FOO, BAR) // expands to "1, 2" (note the "")
 * @endcode
 *
 * @see LEGATE_STRINGIZE_()
 */
#define LEGATE_STRINGIZE(...) LEGATE_STRINGIZE_(__VA_ARGS__)

// Each suffix defines an additional "enabled" state for LEGATE_DEFINED(LEGATE_), i.e. if you
// define
//
// #define LEGATE_DEFINED_ENABLED_FORM_FOO ignored,
//                                  ^^^~~~~~~~~~~~ note suffix
// Results in
//
// #define LEGATE_HAVE_BAR FOO
// LEGATE_DEFINED(LEGATE_HAVE_BAR) // now evalues to 1
#define LEGATE_DEFINED_ENABLED_FORM_1 ignored,
#define LEGATE_DEFINED_ENABLED_FORM_ ignored,

// arguments are either
// - (0, 1, 0, dummy)
// - (1, 0, dummy)
// this final step cherry-picks the middle
#define LEGATE_DEFINED_PRIVATE_3_(ignored, val, ...) val
// the following 2 steps are needed purely for MSVC since it has a nonconforming preprocessor
// and does not expand __VA_ARGS__ in a single step
#define LEGATE_DEFINED_PRIVATE_2_(args) LEGATE_DEFINED_PRIVATE_3_ args
#define LEGATE_DEFINED_PRIVATE_1_(...) LEGATE_DEFINED_PRIVATE_2_((__VA_ARGS__))
// We do not want parentheses around 'x' since we need it to be expanded as-is to push the 1
// forward an arg space
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define LEGATE_DEFINED_PRIVATE(x) LEGATE_DEFINED_PRIVATE_1_(x 1, 0, dummy)

/**
 * @def LEGATE_DEFINED(x)
 *
 * @brief Determine if a preprocessor definition is positively defined.
 *
 * @param x The legate preprocessor definition.
 * @return 1 if the argument is defined and true, 0 otherwise.
 *
 * LEGATE_DEFINED() returns 1 if and only if \a x expands to integer literal 1, or is defined
 * (but empty). In all other cases, LEGATE_DEFINED() returns the integer literal 0. Therefore
 * this macro should not be used if its argument may expand to a non-empty value other than
 * 1. The only exception is if the argument is defined but expands to 0, in which case
 * `LEGATE_DEFINED()` will also expand to 0:
 *
 * @snippet noinit/macros.cc LEGATE_DEFINED
 *
 * Conceptually, `LEGATE_DEFINED()` is equivalent to
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
 * #if LEGATE_DEFINED(FOO_BAR)
 *   foo_bar_is_defined();
 * #else
 *   foo_bar_is_not_defined();
 * #endif
 * @endcode
 *
 * And in regular C++ code:
 *
 * @code
 * if (LEGATE_DEFINED(FOO_BAR)) {
 *   foo_bar_is_defined();
 * } else {
 *   foo_bar_is_not_defined();
 * }
 * @endcode
 *
 * Note that in the C++ example above both arms of the if statement must compile. If this is
 * not desired, then -- since `LEGATE_DEFINED()` produces a compile-time constant expression --
 * the user may use C++17's `if constexpr` to block out one of the arms:
 *
 * @code
 * if constexpr (LEGATE_DEFINED(FOO_BAR)) {
 *   foo_bar_is_defined();
 * } else {
 *   foo_bar_is_not_defined();
 * }
 * @endcode
 *
 * @see LEGATE_CONCAT()
 */
#define LEGATE_DEFINED(x) LEGATE_DEFINED_PRIVATE(LEGATE_CONCAT_(LEGATE_DEFINED_ENABLED_FORM_, x))

/** @} */
