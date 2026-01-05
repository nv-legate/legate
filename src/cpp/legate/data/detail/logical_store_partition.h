/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/partition_placement_info.h>
#include <legate/data/detail/storage.h>
#include <legate/data/detail/storage_partition.h>
#include <legate/data/slice.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/operation/projection.h>
#include <legate/partitioning/detail/restriction.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>

#include <cstdint>
#include <optional>

namespace legate {

class ParallelPolicy;

}  // namespace legate

namespace legate::mapping::detail {

class Machine;

}  // namespace legate::mapping::detail

namespace legate::detail {

class LogicalStorePartition {
 public:
  LogicalStorePartition(InternalSharedPtr<Partition> partition,
                        InternalSharedPtr<StoragePartition> storage_partition,
                        InternalSharedPtr<LogicalStore> store);

  [[nodiscard]] const InternalSharedPtr<Partition>& partition() const;
  [[nodiscard]] const InternalSharedPtr<StoragePartition>& storage_partition() const;
  [[nodiscard]] const InternalSharedPtr<LogicalStore>& store() const;
  [[nodiscard]] InternalSharedPtr<LogicalStore> get_child_store(
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color) const;
  [[nodiscard]] StoreProjection create_store_projection(
    const Domain& launch_domain, const std::optional<SymbolicPoint>& projection = {});
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const;
  [[nodiscard]] Span<const std::uint64_t> color_shape() const;

  /**
   * @brief Gets the partition placement info for this partition.
   *
   * @return The partition placement info for this.
   */
  [[nodiscard]] detail::PartitionPlacementInfo get_placement_info() const;

 private:
  InternalSharedPtr<Partition> partition_{};
  InternalSharedPtr<StoragePartition> storage_partition_{};
  InternalSharedPtr<LogicalStore> store_{};
};

}  // namespace legate::detail

#include <legate/data/detail/logical_store_partition.inl>
