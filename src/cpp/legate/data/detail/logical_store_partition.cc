/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_store_partition.h>

#include <legate_defines.h>

#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/shape.h>
#include <legate/data/detail/transform/shift.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/mapping/detail/machine.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/operation/detail/operation.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/operation/projection.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partitioner.h>
#include <legate/runtime/detail/partition_manager.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/task/detail/task_return_layout.h>
#include <legate/tuning/parallel_policy.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/legion_utilities.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>
#include <legate/utilities/detail/type_traits.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace legate::detail {

InternalSharedPtr<LogicalStore> LogicalStorePartition::get_child_store(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  if (partition_->kind() != Partition::Kind::TILING) {
    throw TracedException<std::runtime_error>{
      "Child stores can be retrieved only from tile partitions"};
  }
  const auto* tiling = static_cast<const Tiling*>(partition_.get());

  if (!tiling->has_color(color)) {
    throw TracedException<std::out_of_range>{
      fmt::format("Color {} is invalid for partition of color shape {}", color, color_shape())};
  }

  auto transform      = store_->transform();
  auto inverted_color = transform->invert_color(std::move(color));
  auto child_storage  = storage_partition_->get_child_storage(storage_partition_, inverted_color);

  auto child_extents = tiling->get_child_extents(store_->extents(), inverted_color);
  auto child_offsets = tiling->get_child_offsets(inverted_color);

  for (auto&& [dim, coff] : legate::detail::enumerate(child_offsets)) {
    if (coff != 0) {
      transform = make_internal_shared<TransformStack>(std::make_unique<Shift>(dim, -coff),
                                                       std::move(transform));
    }
  }

  return make_internal_shared<LogicalStore>(
    std::move(child_extents), std::move(child_storage), store_->type(), std::move(transform));
}

StoreProjection LogicalStorePartition::create_store_projection(
  const Domain& launch_domain, const std::optional<SymbolicPoint>& projection)
{
  if (store_->has_scalar_storage()) {
    return StoreProjection{};
  }

  if (!partition_->has_launch_domain()) {
    return StoreProjection{};
  }

  // We're about to create a legion partition for this store, so the store should have its region
  // created.
  auto legion_partition = storage_partition_->get_legion_partition();
  auto proj_id = store_->compute_projection(launch_domain, partition_->color_shape(), projection);

  return {std::move(legion_partition), proj_id};
}

bool LogicalStorePartition::is_disjoint_for(const Domain& launch_domain) const
{
  return storage_partition_->is_disjoint_for(launch_domain);
}

Span<const std::uint64_t> LogicalStorePartition::color_shape() const
{
  return partition_->color_shape();
}

}  // namespace legate::detail
