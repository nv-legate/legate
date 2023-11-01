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

#include "core/data/detail/store.h"
#include "core/data/shape.h"
#include "core/runtime/detail/field_manager.h"

#include "legion.h"

#include <functional>
#include <memory>
#include <vector>

namespace legate {

struct Partition;
class Tiling;

}  // namespace legate

namespace legate::detail {

class LogicalRegionField : public std::enable_shared_from_this<LogicalRegionField> {
 private:
  friend class FieldManager;

 public:
  LogicalRegionField(FieldManager* manager,
                     const Legion::LogicalRegion& lr,
                     Legion::FieldID fid,
                     std::shared_ptr<LogicalRegionField> parent = nullptr);

  ~LogicalRegionField();

  [[nodiscard]] int32_t dim() const;
  [[nodiscard]] const Legion::LogicalRegion& region() const;
  [[nodiscard]] Legion::FieldID field_id() const;
  [[nodiscard]] const LogicalRegionField& get_root() const;
  [[nodiscard]] Legion::Domain domain() const;

  [[nodiscard]] RegionField map();
  void attach(Legion::PhysicalRegion pr, void* buffer, bool share);
  void detach();
  void allow_out_of_order_destruction();

  [[nodiscard]] std::shared_ptr<LogicalRegionField> get_child(const Tiling* tiling,
                                                              const Shape& color,
                                                              bool complete);
  [[nodiscard]] Legion::LogicalPartition get_legion_partition(const Partition* partition,
                                                              bool complete);

  void add_invalidation_callback(std::function<void()> callback);
  void perform_invalidation_callbacks();

 private:
  FieldManager* manager_{};
  Legion::LogicalRegion lr_{};
  Legion::FieldID fid_{};
  std::shared_ptr<LogicalRegionField> parent_{};
  std::unique_ptr<Legion::PhysicalRegion> pr_{};
  void* attachment_{};
  bool attachment_shared_{};
  bool destroyed_out_of_order_{};
  std::vector<std::function<void()>> callbacks_{};
};

}  // namespace legate::detail

#include "core/data/detail/logical_region_field.inl"
