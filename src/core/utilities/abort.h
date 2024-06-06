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

#include "core/utilities/compiler.h"
#include "core/utilities/macros.h"
#include "core/utilities/typedefs.h"  // for log_legate()

#include <cassert>
#include <cstdlib>  // for std::abort()
#include <nv/target>

namespace legate::comm::coll {

// This is part of the public API, so we cannot change it (even though we probably should)
void collAbort() noexcept;  // NOLINT(readability-identifier-naming)

}  // namespace legate::comm::coll

// Some implementations of assert() don't macro-expand their arguments before stringizing, so
// we enforce that they are via this extra indirection
#define LEGATE_DEVICE_ASSERT_PRIVATE(...) assert(__VA_ARGS__)

#define LEGATE_ABORT(...)                                                                     \
  do {                                                                                        \
    LEGATE_PRAGMA_PUSH();                                                                     \
    LEGATE_PRAGMA_CLANG_IGNORE("-Wgnu-zero-variadic-macro-arguments");                        \
    NV_IF_TARGET(                                                                           \
      NV_IS_HOST,                                                                           \
      (                                                                                     \
        legate::detail::log_legate().error()                                                \
        << "Legate called abort at "  __FILE__  ":" LEGATE_STRINGIZE(__LINE__)  " in "       \
        << __func__ << "(): " << __VA_ARGS__;                                               \
        /* if the collective library has a bespoke abort function, call that first */       \
        legate::comm::coll::collAbort();                                                    \
        /* if we are here, then either the comm library has not been initialized, or it */  \
        /* didn't have an abort mechanism. Either way, we abort normally now. */            \
        std::abort();                                                                       \
      ),                                                                                    \
      (                                                                                     \
        LEGATE_DEVICE_ASSERT_PRIVATE(                                                       \
          0 && "Legate called abort at " __FILE__ ":" LEGATE_STRINGIZE(__LINE__)             \
          " in <unknown device function>: " LEGATE_STRINGIZE(__VA_ARGS__));                  \
      )                                                                                     \
    ) \
    LEGATE_PRAGMA_POP();                                                                      \
  } while (0)
