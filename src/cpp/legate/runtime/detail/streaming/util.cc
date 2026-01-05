/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/streaming/util.h>

#include <legate/utilities/macros.h>
#include <legate/utilities/typedefs.h>

namespace legate::detail {

legate::Logger& log_streaming()
{
  static legate::Logger logger{"legate.streaming"};
  return logger;
}

StreamingErrorContext::StreamingErrorContext(bool strict_mode)
{
  // as an optimization, we disable the error message collection in non-strict
  // streaming mode in non-debug builds. This is because in non-strict mode we do
  // not throw an exception. In debug builds, we collect it to print it to a log
  // regardless of strict mode.
  if constexpr (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    if (!strict_mode) {
      enabled_ = false;
    }
  }
}

}  // namespace legate::detail
