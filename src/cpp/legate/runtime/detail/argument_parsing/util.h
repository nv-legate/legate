/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/span.h>

#include <string>
#include <vector>

namespace legate::detail {

/**
 * @return `true` when Legate is being invoked as a multi-node job, `false` otherwise.
 */
[[nodiscard]] bool multi_node_job();

/**
 * @brief De-duplicate a series of command-line flags, preserving the relative ordering of the
 * flags.
 *
 * Given:
 * ```
 * ["--foo", "--bar", "--baz", "bop", "--foo=1"]
 * ```
 * This routine returns:
 * ```
 * ["--bar", "--baz", "bop", "--foo=1"]
 * ```
 * Note that the relative ordering of arguments is preserved.
 *
 * @param args The arguments to de-duplicate.
 *
 * @return The de-duplicated flags.
 */
[[nodiscard]] std::vector<std::string> deduplicate_command_line_flags(Span<const std::string> args);

}  // namespace legate::detail
