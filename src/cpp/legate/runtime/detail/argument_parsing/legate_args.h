/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/detail/config.h>

namespace legate::detail {

/**
 * @brief Parse `LEGATE_CONFIG` and generate a `Config` database from it.
 *
 * @return The configuration of Legate.
 */
[[nodiscard]] Config handle_legate_args();

}  // namespace legate::detail
