/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace legate::detail {

enum class ArrayKind : std::uint8_t {
  BASE   = 0,
  LIST   = 1,
  STRUCT = 2,
};

}  // namespace legate::detail
