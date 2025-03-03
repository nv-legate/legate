/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/macros.h>

#include <cassert>
#include <nv/target>
#include <sstream>
#include <string_view>

namespace legate::detail {

[[noreturn]] void abort_handler(std::string_view file,
                                std::string_view func,
                                int line,
                                std::stringstream* ss);

#ifndef __has_cpp_attribute
#define __has_cpp_attribute(x) 0
#endif

// The abort handler *really* does not need to optimized for speed in any way, shape, or
// form. The only thing it should be optimized for is compile-time and instruction
// footprint. We already extract the true abort handler to a .cc, but we still need to convert
// all the arguments to string and unfortunately, std::stringstream has a tendency to explode
// (instruction wise) when optimizations are enabled.
#if __has_cpp_attribute(clang::minsize)
#define LEGATE_OPT_MINSIZE [[clang::minsize]]
#elif __has_cpp_attribute(gnu::optimize)
#define LEGATE_OPT_MINSIZE [[gnu::optimize("Os")]]
#else
#define LEGATE_OPT_MINSIZE
#endif

template <typename... T>
[[noreturn]] LEGATE_OPT_MINSIZE void abort_handler_tpl(std::string_view file,
                                                       std::string_view func,
                                                       int line,
                                                       T&&... args)
{
  std::stringstream ss;

  (ss << ... << args);
  legate::detail::abort_handler(file, func, line, &ss);
}

#undef LEGATE_OPT_MINSIZE

}  // namespace legate::detail

// Some implementations of assert() don't macro-expand their arguments before stringizing, so
// we enforce that they are via this extra indirection
#define LEGATE_DEVICE_ASSERT_PRIVATE(...) assert(__VA_ARGS__)

#define LEGATE_ABORT(...)                                                                     \
  do {                                                                                        \
    LEGATE_PRAGMA_PUSH();                                                                     \
    LEGATE_PRAGMA_CLANG_IGNORE("-Wgnu-zero-variadic-macro-arguments");                        \
    NV_IF_TARGET(                                                                             \
      NV_IS_HOST,                                                                             \
      (legate::detail::abort_handler_tpl(__FILE__, __func__, __LINE__, __VA_ARGS__);),        \
      (LEGATE_DEVICE_ASSERT_PRIVATE(                                                          \
         0 && "Legate called abort at " __FILE__ ":" LEGATE_STRINGIZE(                        \
                __LINE__) " in <unknown device function>: " LEGATE_STRINGIZE(__VA_ARGS__));)) \
    LEGATE_PRAGMA_POP();                                                                      \
  } while (0)
