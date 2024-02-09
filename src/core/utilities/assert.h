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

#include "core/utilities/abort.h"
#include "core/utilities/cpp_version.h"
#include "core/utilities/defined.h"

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if LEGATE_CPP_VERSION >= 23
#define LegateAssume(...) [[assume(__VA_ARGS__)]]
#elif defined(_MSC_VER)  // msvc
#define LegateAssume(...) __assume(__VA_ARGS__)
#elif defined(__clang__) && __has_builtin(__builtin_assume)  // clang
#define LegateAssume(...)                             \
  do {                                                \
    _Pragma("clang diagnostic push");                 \
    _Pragma("clang diagnostic ignored \"-Wassume\""); \
    __builtin_assume(__VA_ARGS__);                    \
    _Pragma("clang diagnostic pop");                  \
  } while (0)
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
//   LegateAssume(bar(x) == 2);
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
#define LegateAssume(...)                                      \
  do {                                                         \
    if constexpr (0 && (__VA_ARGS__)) __builtin_unreachable(); \
  } while (0)
#endif

#if __has_builtin(__builtin_expect)
#define LegateLikely(...) __builtin_expect(!!(__VA_ARGS__), 1)
#define LegateUnlikely(...) __builtin_expect(!!(__VA_ARGS__), 0)
#else
#define LegateLikely(...) __VA_ARGS__
#define LegateUnlikely(...) __VA_ARGS__
#endif

#define LegateCheck(...)                                    \
  do {                                                      \
    /* NOLINTNEXTLINE(readability-simplify-boolean-expr) */ \
    if (LegateUnlikely(!(__VA_ARGS__))) {                   \
      LEGATE_ABORT("assertion failed: " << #__VA_ARGS__);   \
    }                                                       \
  } while (0)

#if LegateDefined(LEGATE_USE_DEBUG)
#define LegateAssert(...) LegateCheck(__VA_ARGS__)
#else
#define LegateAssert(...) LegateAssume(__VA_ARGS__)
#endif
