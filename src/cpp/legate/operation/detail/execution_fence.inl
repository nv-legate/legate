/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/execution_fence.h>

namespace legate::detail {

inline ExecutionFence::ExecutionFence(std::uint64_t unique_id, bool block)
  : Operation{unique_id}, block_{block}
{
}

inline Operation::Kind ExecutionFence::kind() const { return Kind::EXECUTION_FENCE; }

}  // namespace legate::detail
