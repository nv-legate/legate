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

#include "core/utilities/typedefs.h"  // for log_legate()

#include <cstdlib>  // for std::abort()

namespace legate::comm::coll {

void collAbort() noexcept;

}  // namespace legate::comm::coll

#define LEGATE_ABORT(...)                                                                     \
  do {                                                                                        \
    legate::detail::log_legate().error()                                                      \
      << "Legate called abort at " << __FILE__ << ':' << __LINE__ << " in " << __func__       \
      << "(): " << __VA_ARGS__;                                                               \
    /* if the collective library has a bespoke abort function, call that first */             \
    legate::comm::coll::collAbort();                                                          \
    /* if we are here, then either the comm library has not been initialized, or it didn't */ \
    /* have an abort mechanism. Either way, we abort normally now. */                         \
    std::abort();                                                                             \
  } while (0)
