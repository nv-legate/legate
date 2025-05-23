/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/detail/config.h>

#include <string_view>

namespace legate::detail {

/**
 * @return Get the value of LEGATE_CONFIG that was parsed.
 */
[[nodiscard]] std::string_view get_parsed_LEGATE_CONFIG();  // NOLINT(readability-identifier-naming)

/**
 * @brief Parse `LEGATE_CONFIG` and generate a `Config` database from it.
 *
 * @return The configuration of Legate.
 */
[[nodiscard]] Config handle_legate_args();

}  // namespace legate::detail
