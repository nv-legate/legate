/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// This file is to be include in every header file in the Legate STL. It
// should be #include'd at the bottom of every file. It should be paired
// with prefix.hpp, which should be #include'd at the top of every file.
//
// INCLUDE GUARDS ARE NOT NEEDED IN THIS HEADER

#include <legate/utilities/macros.h>

#if !LEGATE_DEFINED(LEGATE_STL_DETAIL_PREFIX_INCLUDED)
#error "Did you forget to add prefix.hpp at the top of the file?"
#endif

#undef LEGATE_STL_DETAIL_PREFIX_INCLUDED

#undef requires

#if LEGATE_DEFINED(LEGATE_STL_DETAIL_POP_MACRO_REQUIRES)
#pragma pop_macro("requires")
#undef LEGATE_STL_DETAIL_POP_MACRO_REQUIRES
#endif

LEGATE_PRAGMA_POP();
