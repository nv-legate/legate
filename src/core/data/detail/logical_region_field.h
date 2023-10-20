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

#include <functional>
#include <memory>
#include <unordered_map>

#include "core/data/detail/store.h"
#include "legion.h"

#include "core/data/shape.h"
#include "core/runtime/detail/field_manager.h"

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

 public:
  ~LogicalRegionField();

 public:
  LogicalRegionField(const LogicalRegionField& other)            = default;
  LogicalRegionField& operator=(const LogicalRegionField& other) = default;

 public:
  int32_t dim() const;
  const Legion::LogicalRegion& region() const { return lr_; }
  Legion::FieldID field_id() const { return fid_; }
  const LogicalRegionField& get_root() const;
  Legion::Domain domain() const;

 public:
  RegionField map();
  void attach(Legion::PhysicalRegion pr, void* buffer, bool share);
  void detach();
  void allow_out_of_order_destruction();

 public:
  std::shared_ptr<LogicalRegionField> get_child(const Tiling* tiling,
                                                const Shape& color,
                                                bool complete);
  Legion::LogicalPartition get_legion_partition(const Partition* partition, bool complete);

 public:
  void add_invalidation_callback(std::function<void()> callback);
  void perform_invalidation_callbacks();

 private:
  FieldManager* manager_;
  Legion::LogicalRegion lr_;
  Legion::FieldID fid_;
  std::shared_ptr<LogicalRegionField> parent_;
  std::unique_ptr<Legion::PhysicalRegion> pr_{};
  void* attachment_{nullptr};
  bool attachment_shared_{false};
  bool destroyed_out_of_order_{false};
  std::vector<std::function<void()>> callbacks_{};
};

}  // namespace legate::detail
