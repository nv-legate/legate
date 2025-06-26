/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/copy.h>

namespace legate::detail {

inline Operation::Kind Copy::kind() const { return Kind::COPY; }

inline bool Copy::needs_partitioning() const { return true; }

}  // namespace legate::detail
