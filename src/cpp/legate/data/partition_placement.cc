/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/partition_placement.h>

#include <legate/data/detail/partition_placement.h>
#include <legate/utilities/shared_ptr.h>

namespace legate {

PartitionPlacement::PartitionPlacement(InternalSharedPtr<detail::PartitionPlacement> impl)
  : impl_{std::move(impl)}
{
}

Span<const std::uint64_t> PartitionPlacement::partition_color() const
{
  return impl_->partition_color();
}

std::uint32_t PartitionPlacement::node_id() const { return impl_->node_id(); }

mapping::StoreTarget PartitionPlacement::memory_type() const { return impl_->memory_type(); }

}  // namespace legate
