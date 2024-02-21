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
// should be #include'd at the bottom of every file. It should be paired
// with prefix.hpp, which should be #include'd at the top of every file.
//
// INCLUDE GUARDS ARE NOT NEEDED IN THIS HEADER

#include "core/utilities/defined.h"

#if !LegateDefined(LEGATE_STL_DETAIL_PREFIX_INCLUDED)
#error "Did you forget to add prefix.hpp at the top of the file?"
#endif

#undef LEGATE_STL_DETAIL_PREFIX_INCLUDED

#undef template
#undef requires

#if LegateDefined(LEGATE_STL_DETAIL_POP_MACRO_TEMPLATE)
#pragma pop_macro("template")
#undef LEGATE_STL_DETAIL_POP_MACRO_TEMPLATE
#endif

#if LegateDefined(LEGATE_STL_DETAIL_POP_MACRO_REQUIRES)
#pragma pop_macro("requires")
#undef LEGATE_STL_DETAIL_POP_MACRO_REQUIRES
#endif

LEGATE_STL_PRAGMA_POP()
