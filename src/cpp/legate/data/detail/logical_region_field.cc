/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_region_field.h>

#include <legate/partitioning/detail/partition.h>
#include <legate/runtime/detail/field_manager.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>

#include <fmt/format.h>

#include <stdexcept>

namespace legate::detail {

[[nodiscard]] const Legion::PhysicalRegion& LogicalRegionField::PhysicalState::ensure_mapping(
  const Legion::LogicalRegion& region,
  Legion::FieldID field_id,
  legate::mapping::StoreTarget target)
{
  if (!physical_region().exists()) {
    set_physical_region(Runtime::get_runtime().map_region_field(region, field_id, target));
  } else if (!physical_region().is_mapped()) {
    Runtime::get_runtime().remap_physical_region(physical_region());
  }

  return physical_region();
}

void LogicalRegionField::PhysicalState::set_has_pending_detach()
{
  has_pending_detach_ = attachment_.exists();
}

void LogicalRegionField::PhysicalState::unmap()
{
  // We unmap the field immediately. In the case where a LogicalStore is allowed to be destroyed
  // out-of-order, this unmapping might happen at different times on different shards. Unmapping
  // doesn't go through the Legion pipeline, so from that perspective it's not critical that all
  // shards unmap a region in the same order. The only problematic case is when shard A unmaps
  // region R and shard B doesn't, then both shards launch a task that uses R (or any region
  // that overlaps with R). Then B will unmap/remap around the task, whereas A will not. This
  // shouldn't be an issue in Legate, because once a shard has (locally) freed a root
  // RegionField, there should be no Stores remaining that use it (or any of its sub-regions).
  // Moreover, the field will only start to get reused once all shards have agreed that it's
  // been collected.
  if (physical_region().exists()) {
    Runtime::get_runtime().unmap_physical_region(physical_region());
  }
  set_physical_region(Legion::PhysicalRegion{});
}

void LogicalRegionField::PhysicalState::detach(bool unordered)
{
  if (!attachment().exists()) {
    LEGATE_ASSERT(!has_pending_detach_);
    return;
  }
  attachment_.detach(unordered);
  has_pending_detach_ = false;
}

void LogicalRegionField::PhysicalState::invoke_callbacks()
{
  if (callbacks_.empty()) {
    return;
  }

  for (auto&& callback : callbacks_) {
    callback();
  }
  callbacks_.clear();
}

void LogicalRegionField::PhysicalState::deallocate_attachment(bool wait_on_detach)
{
  if (wait_on_detach && has_pending_detach_) {
    // Needs to flush pending detach operations from the field's previous life.
    // Note that we don't need to progress unordered operations here, because if we're here, that
    // means that the region field is detached in order. (let that sink in :))
    Runtime::get_runtime().flush_scheduling_window();
    LEGATE_ASSERT(!has_pending_detach_);
  }
  // Then, wait until the detach operations are done
  attachment_.maybe_deallocate(wait_on_detach);
}

void LogicalRegionField::PhysicalState::intentionally_leak_physical_region()
{
  if (physical_region_.exists()) {
    static_cast<void>(
      std::make_unique<Legion::PhysicalRegion>(std::move(physical_region_)).release());
  }
}  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)

// ==========================================================================================

LogicalRegionField::~LogicalRegionField() noexcept { release_region_field(); }

std::int32_t LogicalRegionField::dim() const { return lr_.get_dim(); }

const LogicalRegionField& LogicalRegionField::get_root() const
{
  return parent().has_value() ? (*parent())->get_root() : *this;
}

Domain LogicalRegionField::domain() const
{
  return Runtime::get_runtime().get_index_space_domain(lr_.get_index_space());
}

bool LogicalRegionField::is_mapped() const
{
  // Only the root has a physical region at the moment
  if (parent().has_value()) {
    return (*parent())->is_mapped();
  }
  return mapped_;
}

RegionField LogicalRegionField::map(legate::mapping::StoreTarget target)
{
  if (parent().has_value()) {
    LEGATE_ASSERT(!physical_state_->physical_region().exists());
    return (*parent())->map(target);
  }
  set_mapped(true);
  return {dim(),
          region(),
          physical_state_->ensure_mapping(region(), field_id(), target),
          field_id(),
          /*partitioned=*/false};
}

void LogicalRegionField::unmap()
{
  if (parent().has_value()) {
    LEGATE_ASSERT(!physical_state_->physical_region().exists());
    (*parent())->unmap();
    return;
  }
  set_mapped(false);
  physical_state_->unmap();
}

void LogicalRegionField::mark_already_mapped() { mapped_ = true; }

void LogicalRegionField::attach(Legion::PhysicalRegion physical_region,
                                InternalSharedPtr<ExternalAllocation> allocation)
{
  LEGATE_ASSERT(!parent().has_value());
  LEGATE_ASSERT(physical_region.exists());
  // A `is_mapped()` call returns false here for one of two cases:
  //
  // 1. the region field was created with a read-only attachment, or
  // 2. the region field was created with a read-write attachment but has been released
  // (i.e., `release_region_field()` has been invoked).
  //
  // For a live region field, the first condition holds, so the second condition is for case 2.
  LEGATE_ASSERT((physical_region.is_mapped() == is_mapped()) || released_);
  LEGATE_ASSERT(!physical_state_->attachment().exists());
  LEGATE_ASSERT(!physical_state_->physical_region().exists());
  physical_state_->set_attachment(Attachment{physical_region, std::move(allocation)});
  physical_state_->set_physical_region(std::move(physical_region));
}

void LogicalRegionField::attach(Legion::ExternalResources external_resources,
                                std::vector<InternalSharedPtr<ExternalAllocation>> allocations)
{
  LEGATE_ASSERT(!parent().has_value());
  LEGATE_ASSERT(external_resources.exists());
  LEGATE_ASSERT(!physical_state_->attachment().exists());
  physical_state_->set_attachment(
    Attachment{std::move(external_resources), std::move(allocations)});
}

void LogicalRegionField::detach()
{
  auto&& runtime = Runtime::get_runtime();
  if (!runtime.initialized()) {
    physical_state_->intentionally_leak_physical_region();
    return;
  }
  if (parent().has_value()) {
    throw TracedException<std::invalid_argument>{"Manual detach must be called on the root store"};
  }

  // Need to flush the scheduling window to get all pending attach ops to be issued
  runtime.flush_scheduling_window();

  if (!physical_state_->attachment().exists()) {
    throw TracedException<std::invalid_argument>{"Store has no attachment to detach"};
  }
  physical_state_->unmap();
  physical_state_->detach(false /*unordered*/);
  physical_state_->deallocate_attachment();

  // Mark the region field as unmapped as well
  set_mapped(false);
}

void LogicalRegionField::allow_out_of_order_destruction()
{
  if (parent().has_value()) {
    (*parent())->allow_out_of_order_destruction();
  } else {
    destroyed_out_of_order_ = true;
  }
}

void LogicalRegionField::release_region_field() noexcept
{
  if (released_ || parent().has_value()) {
    return;
  }

  // Skip Legion cleanup if non-owning
  if (non_owning_) {
    released_ = true;
    return;
  }

  try {
    // If the runtime is already destroyed, no need to clean up the resource
    if (!has_started()) {
      physical_state_->intentionally_leak_physical_region();
      released_ = true;
      return;
    }

    auto&& runtime = Runtime::get_runtime();

    physical_state_->set_has_pending_detach();
    runtime.issue_release_region_field(physical_state_, destroyed_out_of_order_);

    runtime.field_manager().free_field(
      FreeFieldInfo{shape_, field_size_, lr_, fid_, physical_state_}, destroyed_out_of_order_);
  } catch (const std::exception& exn) {
    LEGATE_ABORT(exn.what());
  }

  set_mapped(false);
  released_ = true;
}

InternalSharedPtr<LogicalRegionField> LogicalRegionField::get_child(const Tiling* tiling,
                                                                    Span<const std::uint64_t> color,
                                                                    bool complete)
{
  auto legion_partition = get_legion_partition(tiling, complete);
  auto color_point      = to_domain_point(color);
  return make_internal_shared<LogicalRegionField>(
    shape_,
    field_size_,
    Runtime::get_runtime().get_subregion(std::move(legion_partition), color_point),
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
  if (parent().has_value()) {
    (*parent())->add_invalidation_callback(std::move(callback));
  } else {
    physical_state_->add_callback(std::move(callback));
  }
}

void LogicalRegionField::perform_invalidation_callbacks()
{
  if (parent().has_value()) {
    // Callbacks should exist only in the root
    (*parent())->perform_invalidation_callbacks();
  } else {
    physical_state_->invoke_callbacks();
  }
}

std::string LogicalRegionField::to_string() const
{
  return fmt::format("({}, ({}, {}), {})",
                     region().get_tree_id(),
                     region().get_index_space().get_id(),
                     region().get_index_space().get_tree_id(),
                     region().get_field_space().get_id());
}

}  // namespace legate::detail
