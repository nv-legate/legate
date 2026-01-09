/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/reduce.h>

namespace legate::detail {

inline void Reduce::validate() {}

inline Operation::Kind Reduce::kind() const { return Kind::REDUCE; }

inline bool Reduce::needs_partitioning() const { return true; }

}  // namespace legate::detail
