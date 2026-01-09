/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/scatter.h>

namespace legate::detail {

inline void Scatter::set_indirect_out_of_range(bool flag) { out_of_range_ = flag; }

inline Operation::Kind Scatter::kind() const { return Kind::SCATTER; }

inline bool Scatter::needs_partitioning() const { return true; }

}  // namespace legate::detail
