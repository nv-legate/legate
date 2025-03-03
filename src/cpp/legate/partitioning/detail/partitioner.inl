/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/partitioner.h>

namespace legate::detail {

inline bool Strategy::parallel(const Operation* op) const { return launch_domain(op).is_valid(); }

// ==========================================================================================

inline Partitioner::Partitioner(Span<const InternalSharedPtr<Operation>> operations)
  : operations_{std::move(operations)}
{
}

}  // namespace legate::detail
