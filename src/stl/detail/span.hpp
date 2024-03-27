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

#include "config.hpp"  // includes <version>

#if __has_include(<span>)
#if defined(__cpp_lib_span) && __cpp_lib_span >= 202002L
#define LEGATE_STL_HAS_STD_SPAN
#endif
#endif

#if LegateDefined(LEGATE_STL_HAS_STD_SPAN)

#include <span>

#else

#define TCB_SPAN_NAMESPACE_NAME std
#include "tcb/span.hpp"
// We define this on purpose so that downstream libs can pretend we have span
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define __cpp_lib_span 1

#endif  // LEGATE_STL_HAS_STD_SPAN
