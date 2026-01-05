/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/logger.h>

namespace legate::detail::comm::coll {

Logger& logger()
{
  static Logger log{"legate.coll"};

  return log;
}

}  // namespace legate::detail::comm::coll
