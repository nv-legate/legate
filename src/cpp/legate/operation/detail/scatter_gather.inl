/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/scatter_gather.h>

namespace legate::detail {

inline void ScatterGather::set_source_indirect_out_of_range(bool flag)
{
  source_indirect_out_of_range_ = flag;
}

inline void ScatterGather::set_target_indirect_out_of_range(bool flag)
{
  target_indirect_out_of_range_ = flag;
}

inline Operation::Kind ScatterGather::kind() const { return Kind::SCATTER_GATHER; }

inline bool ScatterGather::needs_partitioning() const { return true; }

}  // namespace legate::detail
