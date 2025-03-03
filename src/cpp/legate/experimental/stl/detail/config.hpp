/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/macros.h>

/**
 * @cond
 */

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

#define LEGATE_STL_EAT(...)
#define LEGATE_STL_EVAL(M, ...) M(__VA_ARGS__)
#define LEGATE_STL_EXPAND(...) __VA_ARGS__

#define LEGATE_STL_CHECK(...) LEGATE_STL_EXPAND(LEGATE_STL_CHECK_(__VA_ARGS__, 0, ))
// NOLINTNEXTLINE(readability-identifier-naming)
#define LEGATE_STL_CHECK_(XP, NP, ...) NP
#define LEGATE_STL_PROBE(...) LEGATE_STL_PROBE_(__VA_ARGS__, 1)
// NOLINTNEXTLINE(readability-identifier-naming)
#define LEGATE_STL_PROBE_(XP, NP, ...) XP, NP,

////////////////////////////////////////////////////////////////////////////////////////////////////
// Concepts emulation and portability macros.
//   Usage:
//
//   template <typename A, typename B>
//     LEGATE_STL_REQUIRES(Fooable<A> && Barable<B>)
//   void foobar(A a, B b) { ... }
//
#if LEGATE_DEFINED(LEGATE_STL_CONCEPTS())
#define LEGATE_STL_REQUIRES requires
#else
#define LEGATE_STL_REQUIRES LEGATE_STL_EAT
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
// Before clang-16, clang did not like libstdc++'s ranges implementation
#if __has_include(<ranges>) && \
  (defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 201911L) && \
  (!LEGATE_DEFINED(LEGATE_CLANG) || __clang_major__ >= 16 || defined(_LIBCPP_VERSION))
#define LEGATE_STL_HAS_STD_RANGES() 1
#else
#define LEGATE_STL_HAS_STD_RANGES() 0
#endif

#ifndef LEGATE_STL_UNSPECIFIED
#if LEGATE_DEFINED(LEGATE_DOXYGEN)
#define LEGATE_STL_UNSPECIFIED(...) implementation - defined
#else
#define LEGATE_STL_UNSPECIFIED(...) __VA_ARGS__
#endif
#endif

/**
 * @endcond
 */

/**
 * @defgroup stl Legate Standard Template Library
 */

/**
 * @defgroup stl-concepts Concepts
 * @ingroup stl
 */

/**
 * @defgroup stl-containers Containers
 * @ingroup stl
 */

/**
 * @defgroup stl-views Views
 * @ingroup stl
 */

/**
 * @defgroup stl-algorithms Algorithms
 * @ingroup stl
 */

/**
 * @defgroup stl-utilities Utilities
 * @ingroup stl
 */
