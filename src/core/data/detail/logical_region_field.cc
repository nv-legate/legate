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

#include "core/data/detail/logical_region_field.h"
#include "core/partitioning/partition.h"
#include "core/runtime/detail/field_manager.h"
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

LogicalRegionField::LogicalRegionField(FieldManager* manager,
                                       const Legion::LogicalRegion& lr,
                                       Legion::FieldID fid,
                                       std::shared_ptr<LogicalRegionField> parent)
  : manager_(manager), lr_(lr), fid_(fid), parent_(std::move(parent))
{
}

LogicalRegionField::~LogicalRegionField()
{
  // Only free the field once the top-level region is deleted.
  if (parent_ == nullptr) {
    perform_invalidation_callbacks();
    manager_->free_field(lr_, fid_, destroyed_out_of_order_);
  }
}

int32_t LogicalRegionField::dim() const { return lr_.get_dim(); }

const LogicalRegionField& LogicalRegionField::get_root() const
{
  return parent_ != nullptr ? parent_->get_root() : *this;
}

Domain LogicalRegionField::domain() const
{
  return Runtime::get_runtime()->get_index_space_domain(lr_.get_index_space());
}

void LogicalRegionField::allow_out_of_order_destruction()
{
  if (parent_ != nullptr)
    parent_->allow_out_of_order_destruction();
  else
    destroyed_out_of_order_ = true;
}

std::shared_ptr<LogicalRegionField> LogicalRegionField::get_child(const Tiling* tiling,
                                                                  const Shape& color,
                                                                  bool complete)
{
  auto legion_partition = get_legion_partition(tiling, complete);
  auto color_point      = to_domain_point(color);
  return std::make_shared<LogicalRegionField>(
    manager_,
    Runtime::get_runtime()->get_subregion(legion_partition, color_point),
    fid_,
    shared_from_this());
}

Legion::LogicalPartition LogicalRegionField::get_legion_partition(const Partition* partition,
                                                                  bool complete)
{
  return partition->construct(lr_, complete);
}

void LogicalRegionField::add_invalidation_callback(std::function<void()> callback)
{
  if (parent_ != nullptr) {
    parent_->add_invalidation_callback(callback);
  } else {
    callbacks_.push_back(callback);
  }
}

void LogicalRegionField::perform_invalidation_callbacks()
{
  if (parent_ != nullptr) {
#ifdef DEBUG_LEGATE
    // Callbacks should exist only in the root
    assert(callbacks_.empty());
#endif
    parent_->perform_invalidation_callbacks();
  } else {
    for (auto& callback : callbacks_) { callback(); }
    callbacks_.clear();
  }
}

}  // namespace legate::detail
