/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace legate::detail {

class ParsedArgs;

/**
 * @brief Compose the contents of `LEGION_DEFAULT_ARGS`.
 *
 * This routine does not actually set `LEGION_DEFAULT_ARGS`, it only computes what the new
 * value should be.
 *
 * This is technically a private function, but we expose it to test it.
 *
 * @param parsed The parsed command-line arguments.
 *
 * @return The new value of `LEGION_DEFAULT_ARGS`.
 */
[[nodiscard]] std::string compose_legion_default_args(const ParsedArgs& parsed);

/**
 * @brief Configure Legion based on parsed command-line flags.
 *
 * This function sets `LEGION_DEFAULT_ARGS`.
 *
 * @param parsed The parsed command-line arguments.
 */
void configure_legion(const ParsedArgs& parsed);

}  // namespace legate::detail
