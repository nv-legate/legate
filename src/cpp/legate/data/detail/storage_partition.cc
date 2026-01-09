/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/storage_partition.h>

#include <legate_defines.h>

#include <legate/data/detail/shape.h>
#include <legate/mapping/detail/machine.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/tuning/parallel_policy.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/type_traits.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <utility>

namespace legate::detail {

const Storage* StoragePartition::get_root() const { return parent_->get_root(); }

Storage* StoragePartition::get_root() { return parent_->get_root(); }

InternalSharedPtr<const Storage> StoragePartition::get_root(
  const InternalSharedPtr<const StoragePartition>&) const
{
  return parent_->get_root(parent_);
}

InternalSharedPtr<Storage> StoragePartition::get_root(const InternalSharedPtr<StoragePartition>&)
{
  return parent_->get_root(parent_);
}

InternalSharedPtr<Storage> StoragePartition::get_child_storage(
  const InternalSharedPtr<StoragePartition>& self, SmallVector<std::uint64_t, LEGATE_MAX_DIM> color)
{
  LEGATE_ASSERT(self.get() == this);

  if (partition_->kind() != Partition::Kind::TILING) {
    throw TracedException<std::runtime_error>{"Sub-storage is implemented only for tiling"};
  }

  auto tiling        = static_cast<Tiling*>(partition_.get());
  auto child_extents = tiling->get_child_extents(parent_->extents(), color);
  auto child_offsets = tiling->get_child_offsets(color);

  return make_internal_shared<Storage>(
    std::move(child_extents), self, std::move(color), std::move(child_offsets));
}

InternalSharedPtr<LogicalRegionField> StoragePartition::get_child_data(
  Span<const std::uint64_t> color)
{
  if (partition_->kind() != Partition::Kind::TILING) {
    throw TracedException<std::runtime_error>{"Sub-storage is implemented only for tiling"};
  }

  auto tiling = static_cast<Tiling*>(partition_.get());
  return parent_->get_region_field()->get_child(tiling, color, complete_);
}

std::optional<InternalSharedPtr<Partition>> StoragePartition::find_key_partition(
  const mapping::detail::Machine& machine,
  const ParallelPolicy& parallel_policy,
  const Restrictions& restrictions) const
{
  return parent_->find_key_partition(machine, parallel_policy, restrictions);
}

Legion::LogicalPartition StoragePartition::get_legion_partition()
{
  return parent_->get_region_field()->get_legion_partition(partition_.get(), complete_);
}

bool StoragePartition::is_disjoint_for(const Domain& launch_domain) const
{
  return partition_->is_disjoint_for(launch_domain);
}

}  // namespace legate::detail
