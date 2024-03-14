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

#include "core/utilities/defined.h"

#if __has_include(<version>)
#include <version>
#else
#include <ciso646>  // For stdlib feature test macros
#endif

#define LEGATE_STL_DETAIL_CONFIG_INCLUDED

#if defined(__cpp_concepts)
#define LEGATE_STL_CONCEPTS() 1
#else
#define LEGATE_STL_CONCEPTS() 0
#endif

#ifdef __CUDACC__
#define LEGATE_STL_HAS_CUDA() 1
#else
#define LEGATE_STL_HAS_CUDA() 0
#endif

#if defined(__NVCC__)
#define LEGATE_STL_NVCC() 1
#elif defined(__NVCOMPILER)
#define LEGATE_STL_NVHPC() 1
#elif defined(__EDG__)
#define LEGATE_STL_EDG() 1
#elif defined(__clang__)
#define LEGATE_STL_CLANG() 1
#elif defined(__GNUC__)
#define LEGATE_STL_GCC() 1
#elif defined(_MSC_VER)
#define LEGATE_STL_MSVC() 1
#endif

#if !LegateDefined(LEGATE_STL_NVCC())
#define LEGATE_STL_NVCC() 0
#endif
#if !LegateDefined(LEGATE_STL_NVHPC())
#define LEGATE_STL_NVHPC() 0
#endif
#if !LegateDefined(LEGATE_STL_EDG())
#define LEGATE_STL_EDG() 0
#endif
#if !LegateDefined(LEGATE_STL_CLANG())
#define LEGATE_STL_CLANG() 0
#endif
#if !LegateDefined(LEGATE_STL_GCC())
#define LEGATE_STL_GCC() 0
#endif
#if !LegateDefined(LEGATE_STL_MSVC())
#define LEGATE_STL_MSVC() 0
#endif

#define LEGATE_STL_STRINGIZE(...) #__VA_ARGS__

#if LEGATE_STL_NVCC()
#define LEGATE_STL_PRAGMA_PUSH() _Pragma("nv_diagnostic push")
#define LEGATE_STL_PRAGMA_POP() _Pragma("nv_diagnostic pop")
#define LEGATE_STL_PRAGMA_EDG_IGNORE(...) \
  _Pragma(LEGATE_STL_STRINGIZE(nv_diag_suppress __VA_ARGS__))
#elif LEGATE_STL_NVHPC() || LEGATE_STL_EDG()
#define LEGATE_STL_PRAGMA_PUSH() \
  _Pragma("diagnostic push") LEGATE_STL_PRAGMA_EDG_IGNORE(invalid_error_number)
#define LEGATE_STL_PRAGMA_POP() _Pragma("diagnostic pop")
#define LEGATE_STL_PRAGMA_EDG_IGNORE(...) _Pragma(LEGATE_STL_STRINGIZE(diag_suppress __VA_ARGS__))
#elif LEGATE_STL_CLANG() || LEGATE_STL_GCC()
#define LEGATE_STL_PRAGMA_PUSH() _Pragma("GCC diagnostic push")
#define LEGATE_STL_PRAGMA_POP() _Pragma("GCC diagnostic pop")
#define LEGATE_STL_PRAGMA_GNU_IGNORE(...) \
  _Pragma(LEGATE_STL_STRINGIZE(GCC diagnostic ignored __VA_ARGS__))
#else
#define LEGATE_STL_PRAGMA_PUSH()
#define LEGATE_STL_PRAGMA_POP()
#endif

#ifndef LEGATE_STL_PRAGMA_EDG_IGNORE
#define LEGATE_STL_PRAGMA_EDG_IGNORE(...)
#endif
#ifndef LEGATE_STL_PRAGMA_GNU_IGNORE
#define LEGATE_STL_PRAGMA_GNU_IGNORE(...)
#endif

#define LEGATE_STL_CAT_IMPL(A, ...) A##__VA_ARGS__
#define LEGATE_STL_CAT(A, ...) LEGATE_STL_CAT_IMPL(A, __VA_ARGS__)

#define LEGATE_STL_EAT(...)
#define LEGATE_STL_EVAL(M, ...) M(__VA_ARGS__)
#define LEGATE_STL_EXPAND(...) __VA_ARGS__

#define LEGATE_STL_CHECK(...) LEGATE_STL_EXPAND(LEGATE_STL_CHECK_(__VA_ARGS__, 0, ))
#define LEGATE_STL_CHECK_(XP, NP, ...) NP
#define LEGATE_STL_PROBE(...) LEGATE_STL_PROBE_(__VA_ARGS__, 1)
#define LEGATE_STL_PROBE_(XP, NP, ...) XP, NP,

////////////////////////////////////////////////////////////////////////////////////////////////////
// Concepts emulation and portability macros.
//   Usage:
//
//   template <typename A, typename B>
//     LEGATE_STL_REQUIRES(Fooable<A> && Barable<B>)
//   void foobar(A a, B b) { ... }
//
#if LEGATE_STL_CONCEPTS()
#define LEGATE_STL_REQUIRES requires
#else
#define LEGATE_STL_REQUIRES LEGATE_STL_EAT
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// Before clang-16, clang did not like libstdc++'s ranges implementation
#if __has_include(<ranges>) && \
  (defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 201911L) && \
  (!LEGATE_STL_CLANG() || __clang_major__ >= 16 || defined(_LIBCPP_VERSION))
#define LEGATE_STL_HAS_STD_RANGES() 1
#else
#define LEGATE_STL_HAS_STD_RANGES() 0
#endif

#ifndef LEGATE_STL_IMPLEMENTATION_DEFINED
#if LegateDefined(LEGATE_STL_DOXYGEN)
#define LEGATE_STL_IMPLEMENTATION_DEFINED(...) implementation - defined
#else
#define LEGATE_STL_IMPLEMENTATION_DEFINED(...) __VA_ARGS__
#endif
#endif
