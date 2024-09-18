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

#ifndef DOXYGEN

#include "legate_defines.h"

#include "legate/utilities/macros.h"

extern "C" {

// NOLINTBEGIN
const char* __asan_default_options()
{
  constexpr const char* ret =
#include <legate/asan_default_options.h>
    ;
  return ret;
}

const char* __ubsan_default_options()
{
  return
#include <legate/ubsan_default_options.h>
    ;
}

const char* __lsan_default_suppressions()
{
  return
#include <legate/lsan_suppressions.h>
    ;
}

const char* __tsan_default_options()
{
  return
#include <legate/tsan_default_options.h>
    ;
}

const char* __tsan_default_suppressions()
{
  return
#include <legate/tsan_suppressions.h>
    ;
}
// NOLINTEND

}  // extern "C"
#endif
