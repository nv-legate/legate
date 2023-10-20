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
  // Only free associated resources when the top-level region is deleted.
  if (parent_ == nullptr) {
    auto* runtime = Runtime::get_runtime();
    // If the runtime is already destroyed, no need to clean up the resource
    if (!runtime->initialized()) {
      // FIXME: Leak the PhysicalRegion handle if the runtime has already shut down, as
      // there's no hope that this would be collected by the Legion runtime
      pr_.release();
      return;
    }

    perform_invalidation_callbacks();

    // This is a misuse of the Legate API, so it should technically throw an exception, but we
    // shouldn't throw exceptions in destructors, so we just abort.
    if (attachment_shared_) {
      log_legate.error() << "stores created by attaching to a buffer with share=true must be "
                         << "manually detached";
      LEGATE_ABORT;
    }

    // We unmap the field immediately. In the case where a LogicalStore is allowed to be destroyed
    // out-of-order, this unmapping might happen at different times on different shards. Unmapping
    // doesn't go through the Legion pipeline, so from that perspective it's not critical that all
    // shards unmap a region in the same order. The only problematic case is when shard A unmaps
    // region R and shard B doesn't, then both shards launch a task that uses R (or any region that
    // overlaps with R). Then B will unmap/remap around the task, whereas A will not. This shouldn't
    // be an issue in Legate, because once a shard has (locally) freed a root RegionField, there
    // should be no Stores remaining that use it (or any of its sub-regions). Moreover, the field
    // will only start to get reused once all shards have agreed that it's been collected.
    if (pr_ && pr_->is_mapped()) runtime->unmap_physical_region(*pr_);
    Legion::Future can_dealloc =
      (nullptr == attachment_) ? Legion::Future()  // waiting on this is a noop
                               : Runtime::get_runtime()->detach(
                                   *pr_, false /*flush*/, destroyed_out_of_order_ /*unordered*/);
    manager_->free_field(lr_, fid_, can_dealloc, attachment_, destroyed_out_of_order_);
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

RegionField LogicalRegionField::map()
{
  if (parent_ != nullptr) {
    if (LegateDefined(LEGATE_USE_DEBUG)) assert(!pr_);
    return parent_->map();
  }
  if (!pr_) {
    pr_ =
      std::make_unique<Legion::PhysicalRegion>(Runtime::get_runtime()->map_region_field(lr_, fid_));
  } else if (!pr_->is_mapped()) {
    Runtime::get_runtime()->remap_physical_region(*pr_);
  }
  return {dim(), *pr_, fid_};
}

void LogicalRegionField::attach(Legion::PhysicalRegion pr, void* buffer, bool share)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(nullptr == parent_);
    assert(nullptr != buffer && pr.exists());
    assert(nullptr == attachment_ && !pr_);
  }
  pr_                = std::make_unique<Legion::PhysicalRegion>(pr);
  attachment_        = buffer;
  attachment_shared_ = share;
}

void LogicalRegionField::detach()
{
  auto* runtime = Runtime::get_runtime();
  if (!runtime->initialized()) {
    // FIXME: Leak the PhysicalRegion handle if the runtime has already shut down, as
    // there's no hope that this would be collected by the Legion runtime
    pr_.release();
    return;
  }
  if (nullptr != parent_)
    throw std::invalid_argument("Manual detach must be called on the root store");
  if (!attachment_shared_)
    throw std::invalid_argument("Only stores created with share=true can be manually detached");
  assert(nullptr != attachment_ && pr_ && pr_->exists());
  if (pr_->is_mapped()) runtime->unmap_physical_region(*pr_);
  Legion::Future fut = runtime->detach(*pr_, true /*flush*/, false /*unordered*/);
  fut.get_void_result(true /*silence_warnings*/);
  pr_                = nullptr;
  attachment_        = nullptr;
  attachment_shared_ = false;
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
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      // Callbacks should exist only in the root
      assert(callbacks_.empty());
    }
    parent_->perform_invalidation_callbacks();
  } else {
    for (auto& callback : callbacks_) { callback(); }
    callbacks_.clear();
  }
}

}  // namespace legate::detail
