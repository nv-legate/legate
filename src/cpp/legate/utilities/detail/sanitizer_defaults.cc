/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DOXYGEN

#include <legate_defines.h>  // IWYU pragma: keep

#include <legate/utilities/macros.h>  // IWYU pragma: keep

extern "C" {

// NOLINTBEGIN
const char* __asan_default_options()
{
  return
#include <legate/generated/asan_default_options.h>
    ;
}

const char* __ubsan_default_options()
{
  return
#include <legate/generated/ubsan_default_options.h>
    ;
}

const char* __ubsan_default_suppressions()
{
  return
#include <legate/generated/ubsan_suppressions.h>
    ;
}

const char* __lsan_default_suppressions()
{
  return
#include <legate/generated/lsan_suppressions.h>
    ;
}

const char* __tsan_default_options()
{
  return
#include <legate/generated/tsan_default_options.h>
    ;
}

const char* __tsan_default_suppressions()
{
  return
#include <legate/generated/tsan_suppressions.h>
    ;
}
// NOLINTEND

}  // extern "C"
#endif
