/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DOXYGEN

#include <legate_defines.h>

#include <legate/utilities/macros.h>

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
