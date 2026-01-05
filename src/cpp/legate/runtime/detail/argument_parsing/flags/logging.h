/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <string_view>

namespace legate::detail {

/**
 * @brief Convert text-based logging levels to the numeric logging levels that Legion expects.
 *
 * @param log_levels The logging string specification.
 *
 * @return The converted log levels.
 */
[[nodiscard]] std::string convert_log_levels(std::string_view log_levels);

[[nodiscard]] std::string logging_help_str();

}  // namespace legate::detail
