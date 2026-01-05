/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_store_partition.h>

#include <legate_defines.h>

#include <legate/data/detail/partition_placement_info.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/transform/shift.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/operation/projection.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partition/tiling.h>
#include <legate/runtime/detail/projection.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/runtime/detail/shard.h>
#include <legate/type/detail/types.h>
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

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace legate::detail {

InternalSharedPtr<LogicalStore> LogicalStorePartition::get_child_store(
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const
{
  const auto* const tiling = dynamic_cast<const Tiling*>(partition_.get());

  if (!tiling) {
    throw TracedException<std::runtime_error>{
      "Child stores can be retrieved only from tile partitions"};
  }

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

detail::PartitionPlacementInfo LogicalStorePartition::get_placement_info() const
{
  // Algorithm:
  // 1. Get the partition's color shape and launch domain from the underlying partition
  // 2. Compute the projection ID that maps from launch domain points to partition colors
  // 3. Find the corresponding sharding functor that determines which node executes each partition
  // 4. For each point in the launch domain:
  //    a. Use the sharding functor to determine which node (shard) executes this partition
  //    b. Use the projection function to map the launch domain point to a partition color
  //    c. Get the child store for this partition color
  //    d. Determine the memory type where this partition is stored (cached or default SYSMEM)
  //    e. Record the mapping of (partition_color, node_id, memory_type)

  const auto&& color_shape   = partition_->color_shape();
  const auto&& launch_domain = partition_->launch_domain();
  const auto&& projection_id = store_->compute_projection(launch_domain, color_shape);
  const auto&& sharding_id   = find_sharding_functor_by_projection_functor(projection_id);
  Legion::ShardingFunctor* sharding_functor = Legion::Runtime::get_sharding_functor(sharding_id);

  LEGATE_CHECK(sharding_functor != nullptr);

  std::vector<detail::PartitionPlacement> mappings{};
  auto total_nodes               = Runtime::get_runtime().node_count();
  const auto projection_function = find_projection_function(projection_id);

  for (Legion::Domain::DomainPointIterator itr{launch_domain}; itr; itr++) {
    auto assigned_shard = sharding_functor->shard(itr.p, launch_domain, total_nodes);

    const auto&& color_domain_point = projection_function->project_point(itr.p);
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> partition_color;

    partition_color.reserve(color_domain_point.get_dim());
    for (int i = 0; i < color_domain_point.get_dim(); ++i) {
      partition_color.push_back(static_cast<std::uint64_t>(color_domain_point[i]));
    }

    const auto&& child_store = get_child_store(partition_color);

    auto memory_type = [&]() {
      auto mapped_physical_store = child_store->get_cached_physical_store();

      if (mapped_physical_store.has_value()) {
        return mapped_physical_store.value()->target();
      }
      return legate::mapping::StoreTarget::SYSMEM;
    }();

    mappings.emplace_back(std::move(partition_color), assigned_shard, memory_type);
  }

  return detail::PartitionPlacementInfo{std::move(mappings)};
}

}  // namespace legate::detail
