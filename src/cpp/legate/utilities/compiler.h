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

#pragma once

#include "legate/utilities/macros.h"

#include <string>
#include <typeinfo>

#ifdef DOXYGEN
#define LEGATE_DOXYGEN 1
#endif

// Cython does not define a "standard" way of detecting cythonized source compilation, so we
// just check for any one of these macros which I found to be defined in the preamble on my
// machine. We need to check enough of them in case the Cython devs ever decide to change one
// of their names to keep our bases covered.
#if defined(CYTHON_HEX_VERSION) || defined(CYTHON_ABI) || defined(CYTHON_INLINE) ||         \
  defined(CYTHON_RESTRICT) || defined(CYTHON_UNUSED) || defined(CYTHON_USE_CPP_STD_MOVE) || \
  defined(CYTHON_FALLTHROUGH)
#define LEGATE_CYTHON 1
#define LEGATE_CYTHON_DEFAULT_CTOR(class_name) class_name() = default
#else
#define LEGATE_CYTHON 0
#define LEGATE_CYTHON_DEFAULT_CTOR(class_name) static_assert(sizeof(class_name*))
#endif

// The order of these checks is deliberate. Also the fact that they are one unbroken set of if
// -> elif -> endif. For example, clang defines both __clang__ and __GNUC__ so in order to
// detect the actual GCC, we must catch clang first.
#if defined(__NVCC__)
#define LEGATE_NVCC 1
#elif defined(__NVCOMPILER)
#define LEGATE_NVHPC 1
#elif defined(__EDG__)
#define LEGATE_EDG 1
#elif defined(__clang__)
#define LEGATE_CLANG 1
#elif defined(__GNUC__)
#define LEGATE_GCC 1
#elif defined(_MSC_VER)
#define LEGATE_MSVC 1
#endif

#ifndef LEGATE_NVCC
#define LEGATE_NVCC 0
#endif
#ifndef LEGATE_NVHPC
#define LEGATE_NVHPC 0
#endif
#ifndef LEGATE_EDG
#define LEGATE_EDG 0
#endif
#ifndef LEGATE_CLANG
#define LEGATE_CLANG 0
#endif
#ifndef LEGATE_GCC
#define LEGATE_GCC 0
#endif
#ifndef LEGATE_MSVC
#define LEGATE_MSVC 0
#endif

#ifdef __CUDA_ARCH__
#define LEGATE_DEVICE_COMPILE 1
#else
#define LEGATE_DEVICE_COMPILE 0
#endif

#define LEGATE_WPEDANTIC_SEMICOLON_WORKAROUND() \
  static_assert(true, "see https://stackoverflow.com/a/59153563")

#define LEGATE_PRAGMA(...) \
  _Pragma(LEGATE_STRINGIZE_(__VA_ARGS__)) LEGATE_WPEDANTIC_SEMICOLON_WORKAROUND()

#if LEGATE_DEFINED(LEGATE_NVCC)
#define LEGATE_PRAGMA_PUSH() LEGATE_PRAGMA(nv_diagnostic push)
#define LEGATE_PRAGMA_POP() LEGATE_PRAGMA(nv_diagnostic pop)
#define LEGATE_PRAGMA_EDG_IGNORE(...) LEGATE_PRAGMA(nv_diag_suppress __VA_ARGS__)
#elif LEGATE_DEFINED(LEGATE_NVHPC) || LEGATE_DEFINED(LEGATE_EDG)
#define LEGATE_PRAGMA_PUSH() \
  LEGATE_PRAGMA(diagnostic push) LEGATE_PRAGMA_EDG_IGNORE(invalid_error_number)
#define LEGATE_PRAGMA_POP() LEGATE_PRAGMA(diagnostic pop)
#define LEGATE_PRAGMA_EDG_IGNORE(...) LEGATE_PRAGMA(diag_suppress __VA_ARGS__)
#elif LEGATE_DEFINED(LEGATE_CLANG) || LEGATE_DEFINED(LEGATE_GCC)
#define LEGATE_PRAGMA_PUSH() LEGATE_PRAGMA(GCC diagnostic push)
#define LEGATE_PRAGMA_POP() LEGATE_PRAGMA(GCC diagnostic pop)
#define LEGATE_PRAGMA_GNU_IGNORE(...) LEGATE_PRAGMA(GCC diagnostic ignored __VA_ARGS__)
#else
#define LEGATE_PRAGMA_PUSH() LEGATE_WPEDANTIC_SEMICOLON_WORKAROUND()
#define LEGATE_PRAGMA_POP() LEGATE_WPEDANTIC_SEMICOLON_WORKAROUND()
#endif

#ifndef LEGATE_PRAGMA_EDG_IGNORE
#define LEGATE_PRAGMA_EDG_IGNORE(...) LEGATE_WPEDANTIC_SEMICOLON_WORKAROUND()
#endif
#ifndef LEGATE_PRAGMA_GNU_IGNORE
#define LEGATE_PRAGMA_GNU_IGNORE(...) LEGATE_WPEDANTIC_SEMICOLON_WORKAROUND()
#endif

#if LEGATE_DEFINED(LEGATE_GCC) && !LEGATE_DEFINED(LEGATE_CLANG)
#define LEGATE_PRAGMA_GCC_IGNORE(...) LEGATE_PRAGMA_GNU_IGNORE(__VA_ARGS__)
#else
#define LEGATE_PRAGMA_GCC_IGNORE(...) LEGATE_WPEDANTIC_SEMICOLON_WORKAROUND()
#endif

#if LEGATE_DEFINED(LEGATE_CLANG)
#define LEGATE_PRAGMA_CLANG_IGNORE(...) LEGATE_PRAGMA_GNU_IGNORE(__VA_ARGS__)
#else
#define LEGATE_PRAGMA_CLANG_IGNORE(...) LEGATE_WPEDANTIC_SEMICOLON_WORKAROUND()
#endif

#if LEGATE_DEFINED(LEGATE_CLANG) || LEGATE_DEFINED(LEGATE_GCC)
// Don't use LEGATE_PRAGMA here because we can't use LEGATE_WPEDANTIC_SEMICOLON_WORKAROUND for
// it since this must appear after macro uses, which may happen in preprocessor statements
#define LEGATE_DEPRECATED_MACRO_(...) _Pragma(LEGATE_STRINGIZE_(__VA_ARGS__))
#define LEGATE_DEPRECATED_MACRO(...) \
  LEGATE_DEPRECATED_MACRO_(GCC warning LEGATE_STRINGIZE_(This macro is deprecated : __VA_ARGS__))
#else
#define LEGATE_DEPRECATED_MACRO(...)
#endif

namespace legate::detail {

[[nodiscard]] std::string demangle_type(const std::type_info&);

}  // namespace legate::detail
