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

#define LEGATE_STL_COUNT(...) LEGATE_STL_COUNT_IMPL(__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define LEGATE_STL_COUNT_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N

#define LEGATE_STL_COUNT(...) LEGATE_STL_COUNT_IMPL(__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define LEGATE_STL_COUNT_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N

#define LEGATE_STL_FOR_EACH_1(M, _0) M(_0)
#define LEGATE_STL_FOR_EACH_2(M, _0, ...) M(_0) LEGATE_STL_FOR_EACH_1(M, __VA_ARGS__)
#define LEGATE_STL_FOR_EACH_3(M, _0, ...) M(_0) LEGATE_STL_FOR_EACH_2(M, __VA_ARGS__)
#define LEGATE_STL_FOR_EACH_4(M, _0, ...) M(_0) LEGATE_STL_FOR_EACH_3(M, __VA_ARGS__)
#define LEGATE_STL_FOR_EACH_5(M, _0, ...) M(_0) LEGATE_STL_FOR_EACH_4(M, __VA_ARGS__)
#define LEGATE_STL_FOR_EACH_6(M, _0, ...) M(_0) LEGATE_STL_FOR_EACH_5(M, __VA_ARGS__)
#define LEGATE_STL_FOR_EACH_7(M, _0, ...) M(_0) LEGATE_STL_FOR_EACH_6(M, __VA_ARGS__)
#define LEGATE_STL_FOR_EACH_8(M, _0, ...) M(_0) LEGATE_STL_FOR_EACH_7(M, __VA_ARGS__)
#define LEGATE_STL_FOR_EACH_9(M, _0, ...) M(_0) LEGATE_STL_FOR_EACH_8(M, __VA_ARGS__)
#define LEGATE_STL_FOR_EACH(M, ...) \
  LEGATE_STL_CAT(LEGATE_STL_FOR_EACH_, LEGATE_STL_COUNT(__VA_ARGS__))(M, __VA_ARGS__)

////////////////////////////////////////////////////////////////////////////////////////////////////
// Concepts emulation and portability macros.
//   Usage:
//
//   LEGATE_STL_TEMPLATE(class A, class B)
//     LEGATE_STL_REQUIRES([fwd] Fooable<A> && Barable<B>)
//   void foobar(A a, B b) { ... }
//
// The optional `fwd` keyword is used to indicate that this is the definition of a
// constrained function template that has been forward-declared elsewhere.
#define LEGATE_STL_REQUIRES_EAT_fwd
#define LEGATE_STL_REQUIRES_EAT_FWD(...) LEGATE_STL_CAT(LEGATE_STL_REQUIRES_EAT_, __VA_ARGS__)
#define LEGATE_STL_REQUIRES_PROBE(...) LEGATE_STL_PROBE(~)
#define LEGATE_STL_REQUIRES_PROBE_fwd LEGATE_STL_PROBE(~)

#define LEGATE_STL_REQUIRES_PARENS(A, ...) \
  LEGATE_STL_CAT(LEGATE_STL_REQUIRES_PARENS_, LEGATE_STL_CHECK(LEGATE_STL_REQUIRES_PROBE A))
#define LEGATE_STL_REQUIRES_FWD(A, ...)    \
  LEGATE_STL_CAT(LEGATE_STL_REQUIRES_FWD_, \
                 LEGATE_STL_CHECK(LEGATE_STL_CAT(LEGATE_STL_REQUIRES_PROBE_, A)))

#define LEGATE_STL_REQUIRES_PARENS_0(...) LEGATE_STL_REQUIRES_FWD(__VA_ARGS__)(__VA_ARGS__)

#define LEGATE_STL_REQUIRES(...) LEGATE_STL_REQUIRES_PARENS(__VA_ARGS__)(__VA_ARGS__)

#if LEGATE_STL_CONCEPTS()
#define LEGATE_STL_TEMPLATE(...) template <__VA_ARGS__>
#define LEGATE_STL_REQUIRES_PARENS_1(...) requires(__VA_ARGS__)
#define LEGATE_STL_REQUIRES_FWD_0(...) requires(__VA_ARGS__)
#define LEGATE_STL_REQUIRES_FWD_1(...) requires(LEGATE_STL_REQUIRES_EAT_FWD(__VA_ARGS__))
#else
#define LEGATE_STL_TEMPLATE(...) template <__VA_ARGS__,
#define LEGATE_STL_REQUIRES_PARENS_1(...) std::enable_if_t<(__VA_ARGS__), int> Enable = __LINE__ >
#define LEGATE_STL_REQUIRES_FWD_0(...) std::enable_if_t<(__VA_ARGS__), int> Enable = __LINE__ >
#define LEGATE_STL_REQUIRES_FWD_1(...) \
  std::enable_if_t<(LEGATE_STL_REQUIRES_EAT_FWD(__VA_ARGS__)), int> Enable >
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////

#define LEGATE_STL_IIF(BP, TP, FP) LEGATE_STL_CAT(LEGATE_STL_IIF_, BP)(TP, FP)
#define LEGATE_STL_IIF_0(TP, FP) FP
#define LEGATE_STL_IIF_1(TP, FP) TP

////////////////////////////////////////////////////////////////////////////////////////////////////
// For portably declaring attributes on functions and types
//   Usage:
//
//   LEGATE_STL_ATTRIBUTE((attr1, attr2, ...))
//   void foo() { ... }
#define LEGATE_STL_ATTRIBUTE(XP) LEGATE_STL_FOR_EACH(LEGATE_STL_ATTR, LEGATE_STL_EXPAND XP)

// unknown attributes are treated like C++-style attributes
#define LEGATE_STL_ATTR_WHICH_0(ATTR) [[ATTR]]

// custom handling for specific attribute types
#define LEGATE_STL_ATTR_WHICH_1(ATTR) LEGATE_STL_IIF(LEGATE_STL_HAS_CUDA(), __host__, )
#define LEGATE_STL_ATTR_host LEGATE_STL_PROBE(~, 1)
#define LEGATE_STL_ATTR___host__ LEGATE_STL_PROBE(~, 1)

#define LEGATE_STL_ATTR_WHICH_2(ATTR) LEGATE_STL_IIF(LEGATE_STL_HAS_CUDA(), __device__, )
#define LEGATE_STL_ATTR_device LEGATE_STL_PROBE(~, 2)
#define LEGATE_STL_ATTR___device__ LEGATE_STL_PROBE(~, 2)

#define LEGATE_STL_ATTR(ATTR)                                                                      \
  LEGATE_STL_CAT(LEGATE_STL_ATTR_WHICH_, LEGATE_STL_CHECK(LEGATE_STL_CAT(LEGATE_STL_ATTR_, ATTR))) \
  (ATTR)

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
