/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <string_view>

namespace legate::detail {

[[nodiscard]] std::string convert_log_levels(std::string_view log_levels);

[[nodiscard]] std::string logging_help_str();

}  // namespace legate::detail
