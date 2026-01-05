/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace legate::detail {

class ParsedArgs;

/**
 * @brief Configure Realm based on the command-line flags.
 *
 * @param parsed The command-line flags.
 */
void configure_realm(const ParsedArgs& parsed);

}  // namespace legate::detail
