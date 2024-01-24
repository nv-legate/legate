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

#include "core/utilities/cpp_version.h"

#include "legion.h"

#include <cstdlib>

#define LegateConcat_(x, y) x##y
#define LegateConcat(x, y) LegateConcat_(x, y)

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
#define LegateDefined(x) LegateDefinedPrivate(LegateConcat_(LegateDefinedEnabledForm_, x))

namespace legate::comm::coll {

void collAbort() noexcept;

}  // namespace legate::comm::coll

#define LEGATE_ABORT(...)                                                                     \
  do {                                                                                        \
    legate::detail::log_legate().error()                                                      \
      << "Legate called abort at " << __FILE__ << ':' << __LINE__ << " in " << __func__       \
      << "(): " << __VA_ARGS__;                                                               \
    /* if the collective library has a bespoke abort function, call that first */             \
    legate::comm::coll::collAbort();                                                          \
    /* if we are here, then either the comm library has not been initialized, or it didn't */ \
    /* have an abort mechanism. Either way, we abort normally now. */                         \
    std::abort();                                                                             \
  } while (false)

#ifdef __CUDACC__
#define LEGATE_DEVICE_PREFIX __device__
#else
#define LEGATE_DEVICE_PREFIX
#endif

#ifndef LEGION_REDOP_HALF
#error "Legate needs Legion to be compiled with -DLEGION_REDOP_HALF"
#endif

#if !LegateDefined(LEGATE_USE_CUDA)
#ifdef LEGION_USE_CUDA
#define LEGATE_USE_CUDA 1
#endif
#endif

#if !LegateDefined(LEGATE_USE_OPENMP)
#ifdef REALM_USE_OPENMP
#define LEGATE_USE_OPENMP 1
#endif
#endif

#if !LegateDefined(LEGATE_USE_NETWORK)
#if defined(REALM_USE_GASNET1) || defined(REALM_USE_GASNETEX) || defined(REALM_USE_MPI) || \
  defined(REALM_USE_UCX)
#define LEGATE_USE_NETWORK 1
#endif
#endif

#ifdef LEGION_BOUNDS_CHECKS
#define LEGATE_BOUNDS_CHECKS 1
#endif

#define LEGATE_MAX_DIM LEGION_MAX_DIM

// backwards compatibility
#if defined(DEBUG_LEGATE) && !LegateDefined(LEGATE_USE_DEBUG)
#define LEGATE_USE_DEBUG 1
#endif

// TODO(wonchanl): 2022-10-04: Work around a Legion bug, by not instantiating futures on
// framebuffer.
#define LEGATE_NO_FUTURES_ON_FB 1
