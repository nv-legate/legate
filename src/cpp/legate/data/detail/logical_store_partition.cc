/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_store_partition.h>

#include <legate_defines.h>

#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/transform/shift.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partition/tiling.h>
#include <legate/runtime/detail/runtime.h>
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

bool LogicalStorePartition::is_disjoint_for(const Domain& launch_domain) const
{
  return storage_partition_->is_disjoint_for(launch_domain);
}

Span<const std::uint64_t> LogicalStorePartition::color_shape() const
{
  return partition_->color_shape();
}

}  // namespace legate::detail
