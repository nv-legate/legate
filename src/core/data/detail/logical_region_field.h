/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/data/detail/attachment.h"
#include "core/data/detail/external_allocation.h"
#include "core/data/detail/physical_store.h"
#include "core/data/shape.h"
#include "core/runtime/detail/field_manager.h"
#include "core/utilities/internal_shared_ptr.h"

#include "legion.h"

#include <functional>
#include <memory>
#include <vector>

namespace legate {

struct Partition;
class Tiling;

}  // namespace legate

namespace legate::detail {

class LogicalRegionField : public legate::EnableSharedFromThis<LogicalRegionField> {
 private:
  friend class FieldManager;

 public:
  LogicalRegionField(FieldManager* manager,
                     const Legion::LogicalRegion& lr,
                     Legion::FieldID fid,
                     InternalSharedPtr<LogicalRegionField> parent = nullptr);

  ~LogicalRegionField();

  [[nodiscard]] int32_t dim() const;
  [[nodiscard]] const Legion::LogicalRegion& region() const;
  [[nodiscard]] Legion::FieldID field_id() const;
  [[nodiscard]] const LogicalRegionField& get_root() const;
  [[nodiscard]] Legion::Domain domain() const;

  [[nodiscard]] RegionField map();
  void attach(Legion::PhysicalRegion physical_region,
              InternalSharedPtr<ExternalAllocation> allocation);
  void attach(const Legion::ExternalResources& external_resources,
              std::vector<InternalSharedPtr<ExternalAllocation>> allocations);
  void detach();
  void allow_out_of_order_destruction();

  [[nodiscard]] InternalSharedPtr<LogicalRegionField> get_child(const Tiling* tiling,
                                                                const Shape& color,
                                                                bool complete);
  [[nodiscard]] Legion::LogicalPartition get_legion_partition(const Partition* partition,
                                                              bool complete);

  template <typename T>
  void add_invalidation_callback(T&& callback);
  void perform_invalidation_callbacks();

  // Should never copy or move raw logical region field objects
  LogicalRegionField(const LogicalRegionField&)            = delete;
  LogicalRegionField& operator=(const LogicalRegionField&) = delete;
  LogicalRegionField(LogicalRegionField&&)                 = delete;
  LogicalRegionField& operator=(LogicalRegionField&&)      = delete;

 private:
  void add_invalidation_callback_(std::function<void()> callback);

  FieldManager* manager_{};
  Legion::LogicalRegion lr_{};
  Legion::FieldID fid_{};
  InternalSharedPtr<LogicalRegionField> parent_{};
  std::unique_ptr<Legion::PhysicalRegion> pr_{};
  std::unique_ptr<Attachment> attachment_{};
  bool destroyed_out_of_order_{};
  std::vector<std::function<void()>> callbacks_{};
};

}  // namespace legate::detail

#include "core/data/detail/logical_region_field.inl"
