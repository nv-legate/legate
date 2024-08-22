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

#pragma once

#include "legate_defines.h"

#include "core/utilities/macros.h"

extern "C" {

// NOLINTBEGIN
const char* __asan_default_options()
{
  // For whatever reason, stack-use-after-return and leak detection are both stupidly expensive
  // on specifically Linux aarch64, which is used in CI. Basic "hello world" programs take up
  // to 8-9 seconds (!!) to complete with these options enabled.
  //
  // So we disable them.
  return "check_initialization_order=1:"
#if defined(__linux__) && defined(__aarch64__)
         "detect_stack_use_after_return=0:"
#else
         "detect_stack_use_after_return=1:"
#endif
         "alloc_dealloc_mismatch=1:"
         "abort_on_error=1:"
         "strict_string_checks=1:"
         "color=always:"
         "detect_odr_violation=2:"
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
         "protect_shadow_gap=0:"
#endif
#if defined(__linux__) && defined(__aarch64__)
         "detect_leaks=0:"
#endif
         // note trailing ":", this is so that user may write ASAN_OPTIONS+="foo:bar:baz"
         "symbolize=1:";
}

const char* __ubsan_default_options() { return "print_stacktrace=1:"; }

const char* __lsan_default_suppressions()
{
  return "leak:librealm.*\n"
         "leak:liblegion.*\n";
}

const char* __tsan_default_options()
{
  return "halt_on_error=1:"
         "second_deadlock_stack=1:"
         "symbolize=1:"
         "detect_deadlocks=1:";
}

const char* __tsan_default_suppressions()
{
  return "race:Legion::Internal::MemoryManager::create_eager_instance\n"
         "race:Legion::Internal::Operation::perform_registration\n";
}
// NOLINTEND

}  // extern "C"
