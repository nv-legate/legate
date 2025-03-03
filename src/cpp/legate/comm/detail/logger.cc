/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/logger.h>

namespace legate::detail::comm::coll {

Logger& logger()
{
  static Logger log{"coll"};

  return log;
}

}  // namespace legate::detail::comm::coll
