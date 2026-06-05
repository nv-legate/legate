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

bool LogicalStorePartition::is_disjoint_for(const Domain& launch_domain) const
{
  return storage_partition_->is_disjoint_for(launch_domain);
}

Span<const std::uint64_t> LogicalStorePartition::color_shape() const
{
  return partition_->color_shape();
}

}  // namespace legate::detail
