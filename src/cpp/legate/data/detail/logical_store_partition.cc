/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_store_partition.h>

#include <legate/data/detail/logical_store.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

namespace legate::detail {

Span<const std::uint64_t> LogicalStorePartition::color_shape() const
{
  return partition_->color_shape();
}

void LogicalStorePartition::fetch() const { store()->reshuffle(partition_); }

}  // namespace legate::detail
