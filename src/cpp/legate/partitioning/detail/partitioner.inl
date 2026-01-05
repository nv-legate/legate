/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/partitioner.h>

namespace legate::detail {

inline Partitioner::Partitioner(Span<const InternalSharedPtr<Operation>> operations)
  : operations_{operations}
{
}

}  // namespace legate::detail
