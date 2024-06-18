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

#include "core/utilities/macros.h"

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

#if LEGATE_DEFINED(LEGATE_NVCC)
#define LEGATE_PRAGMA_PUSH() _Pragma("nv_diagnostic push")
#define LEGATE_PRAGMA_POP() _Pragma("nv_diagnostic pop")
#define LEGATE_PRAGMA_EDG_IGNORE(...) _Pragma(LEGATE_STRINGIZE_(nv_diag_suppress __VA_ARGS__))
#elif LEGATE_DEFINED(LEGATE_NVHPC) || LEGATE_DEFINED(LEGATE_EDG)
#define LEGATE_PRAGMA_PUSH() \
  _Pragma("diagnostic push") LEGATE_PRAGMA_EDG_IGNORE(invalid_error_number)
#define LEGATE_PRAGMA_POP() _Pragma("diagnostic pop")
#define LEGATE_PRAGMA_EDG_IGNORE(...) _Pragma(LEGATE_STRINGIZE_(diag_suppress __VA_ARGS__))
#elif LEGATE_DEFINED(LEGATE_CLANG) || LEGATE_DEFINED(LEGATE_GCC)
#define LEGATE_PRAGMA_PUSH() _Pragma("GCC diagnostic push")
#define LEGATE_PRAGMA_POP() _Pragma("GCC diagnostic pop")
#define LEGATE_PRAGMA_GNU_IGNORE(...) _Pragma(LEGATE_STRINGIZE_(GCC diagnostic ignored __VA_ARGS__))
#else
#define LEGATE_PRAGMA_PUSH()
#define LEGATE_PRAGMA_POP()
#endif

#ifndef LEGATE_PRAGMA_EDG_IGNORE
#define LEGATE_PRAGMA_EDG_IGNORE(...)
#endif
#ifndef LEGATE_PRAGMA_GNU_IGNORE
#define LEGATE_PRAGMA_GNU_IGNORE(...)
#endif

#if LEGATE_DEFINED(LEGATE_GCC) && !LEGATE_DEFINED(LEGATE_CLANG)
#define LEGATE_PRAGMA_GCC_IGNORE(...) LEGATE_PRAGMA_GNU_IGNORE(__VA_ARGS__)
#else
#define LEGATE_PRAGMA_GCC_IGNORE(...)
#endif

#if LEGATE_DEFINED(LEGATE_CLANG)
#define LEGATE_PRAGMA_CLANG_IGNORE(...) LEGATE_PRAGMA_GNU_IGNORE(__VA_ARGS__)
#else
#define LEGATE_PRAGMA_CLANG_IGNORE(...)
#endif

#if LEGATE_DEFINED(LEGATE_CLANG) || LEGATE_DEFINED(LEGATE_GCC)
#define LEGATE_DEPRECATED_MACRO_(...) _Pragma(LEGATE_STRINGIZE(GCC warning __VA_ARGS__))
#define LEGATE_DEPRECATED_MACRO(...) \
  LEGATE_DEPRECATED_MACRO_("This macro is deprecated: " __VA_ARGS__)
#else
#define LEGATE_DEPRECATED_MACRO(...)
#endif

namespace legate::detail {

[[nodiscard]] std::string demangle_type(const std::type_info&);

}  // namespace legate::detail
