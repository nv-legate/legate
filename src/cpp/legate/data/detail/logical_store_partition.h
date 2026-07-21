/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_store.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>

namespace legate::detail {

class LogicalStorePartition {
 public:
  LogicalStorePartition(InternalSharedPtr<Partition> partition,
                        InternalSharedPtr<LogicalStore> store);

  [[nodiscard]] const InternalSharedPtr<Partition>& partition() const;
  [[nodiscard]] const InternalSharedPtr<LogicalStore>& store() const;
  [[nodiscard]] const Point& color_shape() const;

  void fetch() const;

 private:
  InternalSharedPtr<Partition> partition_{};
  InternalSharedPtr<LogicalStore> store_{};
};

}  // namespace legate::detail

#include <legate/data/detail/logical_store_partition.inl>
