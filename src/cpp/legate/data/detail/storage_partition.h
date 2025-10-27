/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/storage.h>
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

class StoragePartition {
 public:
  StoragePartition(InternalSharedPtr<Storage> parent,
                   InternalSharedPtr<Partition> partition,
                   bool complete);

  [[nodiscard]] const InternalSharedPtr<Partition>& partition() const;
  [[nodiscard]] const Storage* get_root() const;
  [[nodiscard]] Storage* get_root();
  [[nodiscard]] InternalSharedPtr<const Storage> get_root(
    const InternalSharedPtr<const StoragePartition>&) const;
  [[nodiscard]] InternalSharedPtr<Storage> get_root(const InternalSharedPtr<StoragePartition>&);
  [[nodiscard]] InternalSharedPtr<Storage> get_child_storage(
    const InternalSharedPtr<StoragePartition>& self,
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> color);
  [[nodiscard]] InternalSharedPtr<LogicalRegionField> get_child_data(
    Span<const std::uint64_t> color);

  [[nodiscard]] std::optional<InternalSharedPtr<Partition>> find_key_partition(
    const mapping::detail::Machine& machine,
    const ParallelPolicy& parallel_policy,
    const Restrictions& restrictions) const;
  [[nodiscard]] Legion::LogicalPartition get_legion_partition();

  [[nodiscard]] std::int32_t level() const;

  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const;

 private:
  bool complete_{};
  std::int32_t level_{};
  InternalSharedPtr<Storage> parent_{};
  InternalSharedPtr<Partition> partition_{};
};

}  // namespace legate::detail

#include <legate/data/detail/storage_partition.inl>
