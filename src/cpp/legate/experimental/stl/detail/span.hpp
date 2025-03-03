/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/experimental/stl/detail/config.hpp>  // includes <version>
#include <legate/utilities/macros.h>

#if __has_include(<span>)
#if defined(__cpp_lib_span) && __cpp_lib_span >= 202002L
#define LEGATE_STL_HAS_STD_SPAN 1
#endif
#endif

#if LEGATE_DEFINED(LEGATE_STL_HAS_STD_SPAN)

#include <span>

#else

#define TCB_SPAN_NAMESPACE_NAME std
#include <tcb/span.hpp>
// We define this on purpose so that downstream libs can pretend we have span
// NOLINTNEXTLINE
#define __cpp_lib_span 1

#endif  // LEGATE_STL_HAS_STD_SPAN
