/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace legate::detail {

/**
 * @brief Access modes for how operations can access data.
 */
enum class AccessMode : std::uint8_t {
  READ   = 0,
  REDUCE = 1,
  WRITE  = 2,
};

}  // namespace legate::detail
