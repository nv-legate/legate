/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace legate::detail {

template <typename StringType = std::string>
[[nodiscard]] std::vector<StringType> string_split(std::string_view command, char sep = ' ');

/**
 * @return `true` when Legate is being invoked as a multi-node job, `false` otherwise.
 */
[[nodiscard]] bool multi_node_job();

}  // namespace legate::detail
