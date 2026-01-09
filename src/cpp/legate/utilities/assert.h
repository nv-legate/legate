/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>  // LEGATE_USE_DEBUG

#include <legate/utilities/abort.h>
#include <legate/utilities/cpp_version.h>
#include <legate/utilities/macros.h>

#if LEGATE_CPP_VERSION >= 23
#include <utility>

#define LEGATE_UNREACHABLE() ::std::unreachable()
#elif __has_builtin(__builtin_unreachable) || defined(__GNUC__)  // clang, gcc
#define LEGATE_UNREACHABLE() __builtin_unreachable()
#elif LEGATE_DEFINED(LEGATE_MSVC)  // MSVC
#define LEGATE_UNREACHABLE() __assume(false)
#else
#define LEGATE_UNREACHABLE() LEGATE_ABORT("Unreachable code path executed!")
#endif

#if LEGATE_CPP_VERSION >= 23
#define LEGATE_ASSUME(...) [[assume(__VA_ARGS__)]]
#elif LEGATE_DEFINED(LEGATE_MSVC)
#define LEGATE_ASSUME(...) __assume(__VA_ARGS__)
#elif LEGATE_DEFINED(LEGATE_CLANG) && __has_builtin(__builtin_assume)
#define LEGATE_ASSUME(...)                                  \
  do {                                                      \
    LEGATE_PRAGMA_PUSH();                                   \
    LEGATE_PRAGMA_GNU_IGNORE("-Wassume");                   \
    /* NOLINTNEXTLINE(readability-simplify-boolean-expr) */ \
    __builtin_assume(!!(__VA_ARGS__));                      \
    LEGATE_PRAGMA_POP();                                    \
  } while (0)
#elif LEGATE_DEFINED(LEGATE_GCC) && (__GNUC__ >= 13)
#define LEGATE_ASSUME(...) __attribute__((__assume__(__VA_ARGS__)))
#else  // gcc (and really old clang)
// gcc does not have its own __builtin_assume() intrinsic. One could fake it via
//
// if (!cond) __builtin_unreachable();
//
// but this it unsavory because the side effects of cond are not guaranteed to be discarded. In
// most circumstances gcc will optimize out the if (because any evaluation for which cond is
// false is ostensibly unreachable, and that results in undefined behavior anyway). But it
// cannot always do so. This is especially the case for opaque or non-inline function calls:
//
// extern int bar(int);
//
// int foo(int x) {
//   LEGATE_ASSUME(bar(x) == 2);
//   if (bar(x) == 2) {
//     return 1;
//   } else {
//     return 0;
//   }
// }
//
// Here gcc would (if just using the plain 'if' version) emit 2 calls to bar(). But since we
// elide the branch at compile-time, our version doesn't have this problem. Note we still have
// cond "tested" in the condition, but this is done to silence unused-but-set variable warnings
#define LEGATE_ASSUME(...)               \
  do {                                   \
    if constexpr (0 && !(__VA_ARGS__)) { \
      LEGATE_UNREACHABLE();              \
    }                                    \
  } while (0)
#endif

#if __has_builtin(__builtin_expect) || defined(__GNUC__)
#define LEGATE_LIKELY(...) __builtin_expect(!!(__VA_ARGS__), 1)
#define LEGATE_UNLIKELY(...) __builtin_expect(!!(__VA_ARGS__), 0)
#else
#define LEGATE_LIKELY(...) __VA_ARGS__
#define LEGATE_UNLIKELY(...) __VA_ARGS__
#endif

#define LEGATE_CHECK(...)                                   \
  do {                                                      \
    /* NOLINTNEXTLINE(readability-simplify-boolean-expr) */ \
    if (LEGATE_UNLIKELY(!(__VA_ARGS__))) {                  \
      LEGATE_ABORT("assertion failed: " #__VA_ARGS__);      \
    }                                                       \
  } while (0)

#if LEGATE_DEFINED(LEGATE_USE_DEBUG)
#define LEGATE_ASSERT(...) LEGATE_CHECK(__VA_ARGS__)
#else
#define LEGATE_ASSERT(...) LEGATE_ASSUME(__VA_ARGS__)
#endif
