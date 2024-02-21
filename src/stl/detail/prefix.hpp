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

// This file is to be include in every header file in the Legate STL. It
// should be the last #include before the first line of actual C++ code.
// It should be paired with suffix.hpp, which should be last in the file.
//
// INCLUDE GUARDS ARE NOT NEEDED IN THIS HEADER

#include "core/utilities/defined.h"

#if !LegateDefined(LEGATE_STL_DETAIL_CONFIG_INCLUDED)
#error "config.hpp must be included before prefix.hpp"
#endif

#if LegateDefined(LEGATE_STL_DETAIL_PREFIX_INCLUDED)
#error "prefix.hpp included twice. Did you forget suffix.hpp elsewhere?"
#endif

#define LEGATE_STL_DETAIL_PREFIX_INCLUDED

#ifdef template
#define LEGATE_STL_DETAIL_POP_MACRO_TEMPLATE
#pragma push_macro("template")
#undef template
#endif

#ifdef requires
#define LEGATE_STL_DETAIL_POP_MACRO_REQUIRES
#pragma push_macro("requires")
#undef requires
#endif

#define template(...) LEGATE_STL_TEMPLATE(__VA_ARGS__)

#if LEGATE_STL_CONCEPTS()
#define requires requires
#else
#define requires LEGATE_STL_REQUIRES
#endif

LEGATE_STL_PRAGMA_PUSH()
LEGATE_STL_PRAGMA_EDG_IGNORE(737)  // using-declaration ignored; it refers to the current namespace
LEGATE_STL_PRAGMA_EDG_IGNORE(20011)  // calling a __host__ function [...] from a __host__ __device__
                                     // function is not allowed
LEGATE_STL_PRAGMA_EDG_IGNORE(20012)  // __host__ annotation is ignored on a function[...] that is
                                     // explicitly defaulted on its first declaration
LEGATE_STL_PRAGMA_EDG_IGNORE(20014)  // calling a __host__ function [...] from a __host__ __device__
                                     // function is not allowed
