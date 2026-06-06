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

bool LogicalStorePartition::is_disjoint_for(const Domain& launch_domain) const
{
  return partition_->is_disjoint_for(launch_domain);
}

Span<const std::uint64_t> LogicalStorePartition::color_shape() const
{
  return partition_->color_shape();
}

}  // namespace legate::detail
