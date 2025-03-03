/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/field_manager.h>

#include <legate/data/detail/logical_region_field.h>
#include <legate/runtime/detail/config.h>
#include <legate/runtime/detail/region_manager.h>
#include <legate/runtime/detail/runtime.h>

namespace legate::detail {

FieldManager::~FieldManager()
{
  // We are shutting down, so just free all the buffer copies we've made to attach to, without
  // waiting on the detachments to finish.
  for (auto&& [_, queue] : ordered_free_fields_) {
    for (; !queue.empty(); queue.pop()) {
      queue.front().state->deallocate_attachment();
    }
  }
}

InternalSharedPtr<LogicalRegionField> FieldManager::allocate_field(InternalSharedPtr<Shape> shape,
                                                                   std::uint32_t field_size)
{
  auto result = try_reuse_field_(shape, field_size);
  if (result != nullptr) {
    return result;
  }
  return create_new_field_(std::move(shape), field_size);
}

InternalSharedPtr<LogicalRegionField> FieldManager::import_field(InternalSharedPtr<Shape> shape,
                                                                 std::uint32_t field_size,
                                                                 Legion::LogicalRegion region,
                                                                 Legion::FieldID field_id)
{
  return make_internal_shared<LogicalRegionField>(
    std::move(shape), field_size, std::move(region), field_id);
}

void FieldManager::free_field(FreeFieldInfo info, bool /*unordered*/)
{
  if (log_legate().want_debug()) {
    log_legate().debug() << "Field " << info.field_id << " on region " << info.region
                         << " freed in-order";
  }
  Runtime::get_runtime()->issue_discard_field(info.region, info.field_id);
  if (info.shape->ready()) {
    auto& queue = ordered_free_fields_[OrderedQueueKey{info.shape->index_space(), info.field_size}];
    queue.emplace(std::move(info));
  } else {
    // TODO(mpapadakis): This is an output store that was collected before its shape was requested.
    // We're not going to block to find out its shape, so we don't know under which free-field queue
    // to place it. For now we just leak its handle (its memory should have already been collected
    // due to the discard_field above). If we want to try to reuse it, we should cache it separately
    // and re-evaluate once its shape becomes known. At the very least we should delete the
    // IndexSpace, to prevent accumulation of Legion handles.
  }
}

InternalSharedPtr<LogicalRegionField> FieldManager::try_reuse_field_(
  const InternalSharedPtr<Shape>& shape, std::uint32_t field_size)
{
  if (!shape->ready()) {
    // We are being asked to make a field for an unbound output store. We won't know the store's
    // size until its producing task has executed, so naturally we can't safely reuse any of our
    // existing fields for it.
    return nullptr;
  }
  auto& queue = ordered_free_fields_[OrderedQueueKey{shape->index_space(), field_size}];
  if (queue.empty()) {
    return nullptr;
  }
  const auto& info = queue.front();
  info.state->deallocate_attachment();
  auto rf = make_internal_shared<LogicalRegionField>(shape, field_size, info.region, info.field_id);
  if (log_legate().want_debug()) {
    log_legate().debug() << "Field " << info.field_id << " on region " << info.region
                         << " recycled for shape " << shape->to_string() << " field size "
                         << info.field_size;
  }
  queue.pop();
  return rf;
}

InternalSharedPtr<LogicalRegionField> FieldManager::create_new_field_(
  InternalSharedPtr<Shape> shape, std::uint32_t field_size)
{
  auto* rgn_mgr  = Runtime::get_runtime()->find_or_create_region_manager(shape->index_space());
  auto [lr, fid] = rgn_mgr->allocate_field(field_size);
  if (log_legate().want_debug()) {
    log_legate().debug() << "Field " << fid << " created on region " << lr << " for shape "
                         << shape->to_string() << " field size " << field_size;
  }
  return make_internal_shared<LogicalRegionField>(std::move(shape), field_size, std::move(lr), fid);
}

// ==========================================================================================

ConsensusMatchingFieldManager::~ConsensusMatchingFieldManager()
{
  // We are shutting down, so just free all the buffer copies we've made to attach to, without
  // waiting on the detachments to finish.
  for (auto&& [_, info] : info_for_match_items_) {
    info.state->deallocate_attachment(false);
  }
  for (auto&& info : unordered_free_fields_) {
    info.state->deallocate_attachment(false);
  }
}

InternalSharedPtr<LogicalRegionField> ConsensusMatchingFieldManager::allocate_field(
  InternalSharedPtr<Shape> shape, std::uint32_t field_size)
{
  maybe_issue_field_match_(shape, field_size);
  // If there's a field that every shard is guaranteed to have, re-use that.
  if (auto result = try_reuse_field_(shape, field_size)) {
    return result;
  }
  // If there's an outstanding consensus match, block on it in case it frees any fields we can use.
  if (outstanding_match_.has_value()) {
    process_outstanding_match_();
    if (auto result = try_reuse_field_(shape, field_size)) {
      return result;
    }
  }
  // Otherwise create a fresh field.
  return create_new_field_(std::move(shape), field_size);
}

InternalSharedPtr<LogicalRegionField> ConsensusMatchingFieldManager::import_field(
  InternalSharedPtr<Shape> shape,
  std::uint32_t field_size,
  Legion::LogicalRegion region,
  Legion::FieldID field_id)
{
  // We didn't allocate this particular field, it will be produced as the unbound output of a task.
  // Still we need to bump up our credit count, because that field then becomes part of our set of
  // managed fields, and we want it to be freed in due time.
  maybe_issue_field_match_(shape, field_size);
  return make_internal_shared<LogicalRegionField>(
    std::move(shape), field_size, std::move(region), field_id);
}

void ConsensusMatchingFieldManager::free_field(FreeFieldInfo info, bool unordered)
{
  if (unordered) {
    if (log_legate().want_debug()) {
      log_legate().debug() << "Field " << info.field_id << " on region " << info.region
                           << " freed locally";
    }
    unordered_free_fields_.emplace_back(std::move(info));
  } else {
    FieldManager::free_field(std::move(info), unordered);
  }
}

std::uint32_t ConsensusMatchingFieldManager::calculate_match_credit_(
  const InternalSharedPtr<Shape>& shape, std::uint32_t field_size) const
{
  if (!shape->ready()) {
    // We don't know how big an unbound output field is going to be (until its producing task has
    // completed), therefore make a worst-case assumption.
    return Runtime::get_runtime()->field_reuse_freq();
  }
  const auto size             = shape->volume() * field_size;
  const auto field_reuse_size = Runtime::get_runtime()->field_reuse_size();
  if (size > field_reuse_size) {
    LEGATE_CHECK(field_reuse_size > 0);
    return (size + field_reuse_size - 1) / field_reuse_size;
  }
  return 1;
}

void ConsensusMatchingFieldManager::maybe_issue_field_match_(const InternalSharedPtr<Shape>& shape,
                                                             std::uint32_t field_size)
{
  // Here we keep track of the total size of fields we've allocated since the last consensus match
  // we emitted. Every time our total goes over a threshold, we first block on the previous
  // consensus match, process the result so we can free some memory, then asynchronously emit
  // another one with the current set of locally-freed fields. This approach should keep the amount
  // of "wasted" space (instances that we can't discard because, even though we have freed the
  // corresponding fields locally, we don't know that all nodes have freed them) somewhat bounded.
  field_match_counter_ += calculate_match_credit_(shape, field_size);
  if (field_match_counter_ >= Runtime::get_runtime()->field_reuse_freq()) {
    process_outstanding_match_();
    issue_field_match_();
    field_match_counter_ = 0;
  }
}

void ConsensusMatchingFieldManager::issue_field_match_()
{
  LEGATE_ASSERT(!outstanding_match_.has_value());
  LEGATE_ASSERT(info_for_match_items_.empty());
  // Check if there are any freed fields that are shared across all the shards. We have to test this
  // deterministically no matter what, even if we don't have any fields to offer ourselves, because
  // this is a collective with other shards. We need to separately record the full information for
  // each item taking part in the match, because the actual values we match between ranks only
  // include a subset of this information.
  std::vector<MatchItem> input;

  input.reserve(unordered_free_fields_.size());
  for (auto&& info : unordered_free_fields_) {
    auto&& item                 = input.emplace_back(info.region.get_tree_id(), info.field_id);
    info_for_match_items_[item] = std::move(info);
  }
  LEGATE_CHECK(info_for_match_items_.size() == unordered_free_fields_.size());
  unordered_free_fields_.clear();
  log_legate().debug() << "Consensus match emitted with " << input.size() << " local fields";
  // Dispatch the match and put it on the queue of outstanding matches, but don't block on it yet.
  outstanding_match_ = Runtime::get_runtime()->issue_consensus_match(std::move(input));
}

void ConsensusMatchingFieldManager::process_outstanding_match_()
{
  if (!outstanding_match_.has_value()) {
    return;
  }
  const auto runtime = Runtime::get_runtime();
  // We need to flush the window before we ask the runtime to progress unordered operations, to make
  // sure any pending ReleaseRegionField operations that actually unmap/detach fields that we're
  // about to match on have been emitted. If we were to match on a field whose ReleaseRegionField is
  // still in the queue, we run the risk that the field's (unordered) unmap/detach operation gets
  // emitted later and never inserted into the Legion task stream. If we then block on this
  // unmap/detach that has not been explicitly "progressed", the Legion runtime would hang (the
  // runtime doesn't "reap" pending unordered items automatically). We do this here before we block
  // on the consensus match.
  runtime->flush_scheduling_window();
  outstanding_match_->wait();
  log_legate().debug() << "Consensus match result: " << outstanding_match_->output().size() << "/"
                       << outstanding_match_->input().size() << " fields matched";
  // Ask the runtime to find all unordered operations that have been observed on all shards, and
  // dispatch them right now. This will cover any unordered detachments on fields that all shards
  // just matched on; if a LogicalRegionField has been destroyed on all shards, that means its
  // detachment has also been emitted on all shards. This makes it safe to later block on any of
  // those detachments, since they are all guaranteed to be in the task stream now, and will
  // eventually complete.
  runtime->progress_unordered_operations();
  // Put all the matched fields into the ordered queue, in the same order as the match result,
  // which is the same order that all shards will see.
  for (auto&& item : outstanding_match_->output()) {
    auto it = info_for_match_items_.find(item);
    LEGATE_CHECK(it != info_for_match_items_.end());
    FieldManager::free_field(std::move(it->second), false /*unordered*/);
    info_for_match_items_.erase(it);
  }
  // All fields that weren't matched can go back into the unordered queue, to be included in the
  // next consensus match that we run.
  unordered_free_fields_.reserve(unordered_free_fields_.size() + info_for_match_items_.size());
  for (auto&& [_, info] : info_for_match_items_) {
    unordered_free_fields_.emplace_back(std::move(info));
  }
  outstanding_match_.reset();
  info_for_match_items_.clear();
}

}  // namespace legate::detail
