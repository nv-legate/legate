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
 * @brief Determine the number of ranks in use based on environment variables
 * OMPI_COMM_WORLD_SIZE, PMI_SIZE, MV2_COMM_WORLD_SIZE, and SLURM_NTASKS.
 *
 * @return the number of ranks in use
 */
[[nodiscard]] std::uint32_t num_ranks();

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

/**
 * @brief Determine whether GDS is available and usable on the system.
 *
 * This routine is a guess. It does not return reliable answers because there is no way to know
 * if GDS will work until you try to use it and it either fails or succeeds. This routine tries
 * its level best to guess based on the ability to load the cuFile driver, the existence of
 * Linux-specific file drivers and other black magic.
 *
 * @return `true` if GDS is available, `false` if not.
 */
[[nodiscard]] bool is_gds_maybe_available();

}  // namespace legate::detail
