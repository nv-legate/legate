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

#include "core/runtime/detail/field_manager.h"

#include "core/data/detail/logical_region_field.h"
#include "core/runtime/detail/region_manager.h"
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

FieldManager::FieldManager(Runtime* runtime, const Domain& shape, uint32_t field_size)
  : runtime_{runtime}, shape_{shape}, field_size_{field_size}
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    std::stringstream ss;
    if (shape.is_valid()) {
      ss << shape;
    } else {
      ss << "()";
    }
    log_legate().debug() << "Field manager " << this << " created for shape " << std::move(ss).str()
                         << ", field_size " << field_size;
  }
}

FieldManager::~FieldManager()
{
  // We are shutting down, so just free all the buffer copies we've made to attach to, without
  // waiting on the detachments to finish.
  for (const FreeFieldInfo& info : ordered_free_fields_) {
    if (info.attachment) {
      info.attachment->maybe_deallocate();
    }
  }
}

InternalSharedPtr<LogicalRegionField> FieldManager::allocate_field()
{
  auto result = try_reuse_field();
  if (result != nullptr) {
    return result;
  }
  // If there are no more field matches to process, then we completely failed to reuse a field.
  return create_new_field();
}

InternalSharedPtr<LogicalRegionField> FieldManager::import_field(
  const Legion::LogicalRegion& region, Legion::FieldID field_id)
{
  // Import the region only if the region manager is created fresh
  auto rgn_mgr = runtime_->find_or_create_region_manager(shape_);

  if (!rgn_mgr->has_space()) {
    rgn_mgr->import_region(region);
  }
  log_legate().debug("Field %u imported in field manager %p", field_id, static_cast<void*>(this));
  return make_internal_shared<LogicalRegionField>(this, region, field_id);
}

void FieldManager::free_field(const Legion::LogicalRegion& region,
                              Legion::FieldID field_id,
                              Legion::Future can_dealloc,
                              std::unique_ptr<Attachment> attachment,
                              bool /*unordered*/)
{
  log_legate().debug(
    "Field %u freed in-order in field manager %p", field_id, static_cast<void*>(this));
  ordered_free_fields_.emplace_back(
    region, field_id, std::move(can_dealloc), std::move(attachment));
}

InternalSharedPtr<LogicalRegionField> FieldManager::try_reuse_field()
{
  if (ordered_free_fields_.empty()) {
    return nullptr;
  }
  const auto& info = ordered_free_fields_.front();
  if (info.attachment) {
    info.can_dealloc.get_void_result(true /*silence_warnings*/);
    info.attachment->maybe_deallocate();
  }
  auto* rf = new LogicalRegionField(this, info.region, info.field_id);
  log_legate().debug(
    "Field %u recycled in field manager %p", info.field_id, static_cast<void*>(this));
  ordered_free_fields_.pop_front();
  return InternalSharedPtr<LogicalRegionField>{rf};
}

InternalSharedPtr<LogicalRegionField> FieldManager::create_new_field()
{
  // If there are no more field matches to process, then we completely failed to reuse a field.
  auto rgn_mgr   = runtime_->find_or_create_region_manager(shape_);
  auto [lr, fid] = rgn_mgr->allocate_field(field_size_);
  auto* rf       = new LogicalRegionField{this, lr, fid};

  log_legate().debug("Field %u created in field manager %p", fid, static_cast<void*>(this));
  return InternalSharedPtr<LogicalRegionField>{rf};
}

// ==========================================================================================

ConsensusMatchingFieldManager::ConsensusMatchingFieldManager(Runtime* runtime,
                                                             const Domain& shape,
                                                             uint32_t field_size)
  : FieldManager{runtime, shape, field_size}
{
  auto size = shape.dim == 0 ? 1 : (shape.get_volume() * field_size);
  if (size > Config::max_field_reuse_size) {
    assert(Config::max_field_reuse_size > 0);
    field_match_credit_ = (size + Config::max_field_reuse_size - 1) / Config::max_field_reuse_size;
  }
}

ConsensusMatchingFieldManager::~ConsensusMatchingFieldManager()
{
  // We are shutting down, so just free all the buffer copies we've made to attach to, without
  // waiting on the detachments to finish.
  for (auto& infos : info_for_match_items_) {
    for (auto& [item, info] : infos) {
      if (info.attachment) {
        info.attachment->maybe_deallocate();
      }
    }
  }
  for (const FreeFieldInfo& info : unordered_free_fields_) {
    if (info.attachment) {
      info.attachment->maybe_deallocate();
    }
  }
}

InternalSharedPtr<LogicalRegionField> ConsensusMatchingFieldManager::allocate_field()
{
  issue_field_match();
  while (!ordered_free_fields_.empty() || !matches_.empty()) {
    auto result = try_reuse_field();
    // If there's a field that every shard is guaranteed to have, re-use that.
    if (result != nullptr) {
      return result;
    }
    if (matches_.empty()) {
      break;
    }
    // If there are any field matches we haven't processed yet, process the next one, then go back
    // and check if any fields were just added to the "ordered" queue.
    process_next_field_match();
  }
  return create_new_field();
}

void ConsensusMatchingFieldManager::free_field(const Legion::LogicalRegion& region,
                                               Legion::FieldID field_id,
                                               Legion::Future can_dealloc,
                                               std::unique_ptr<Attachment> attachment,
                                               bool unordered)
{
  if (unordered) {
    log_legate().debug(
      "Field %u freed locally in field manager %p", field_id, static_cast<void*>(this));
    unordered_free_fields_.emplace_back(
      region, field_id, std::move(can_dealloc), std::move(attachment));
  } else {
    FieldManager::free_field(
      region, field_id, std::move(can_dealloc), std::move(attachment), unordered);
  }
}

void ConsensusMatchingFieldManager::issue_field_match()
{
  // Check if there are any freed fields that are shared across all the shards. We have to
  // test this deterministically no matter what, even if we don't have any fields to offer
  // ourselves, because this is a collective with other shards.
  field_match_counter_ += field_match_credit_;
  if (field_match_counter_ < runtime_->field_reuse_freq()) {
    return;
  }
  field_match_counter_ = 0;
  // We need to separately record the region that corresponds to each item in this match, because
  // the match itself only uses a subset of the full region info.
  auto& infos = info_for_match_items_.emplace_back();
  std::vector<MatchItem> input;

  input.reserve(unordered_free_fields_.size());
  for (auto& info : unordered_free_fields_) {
    auto&& item = input.emplace_back(info.region.get_tree_id(), info.field_id);
    infos[item] = std::move(info);
  }
  assert(infos.size() == unordered_free_fields_.size());
  unordered_free_fields_.clear();
  // Dispatch the match and put it on the queue of outstanding matches, but don't block on it yet.
  // We'll do that when we run out of ordered fields.
  matches_.push_back(runtime_->issue_consensus_match(std::move(input)));
  log_legate().debug("Consensus match emitted with %zu local fields in field manager %p",
                     infos.size(),
                     static_cast<void*>(this));
}

void ConsensusMatchingFieldManager::process_next_field_match()
{
  assert(!matches_.empty());
  auto& match = matches_.front();
  auto& infos = info_for_match_items_.front();
  match.wait();
  log_legate().debug("Consensus match result in field manager %p: %zu/%zu fields matched",
                     static_cast<void*>(this),
                     match.output().size(),
                     match.input().size());
  // Ask the runtime to find all unordered operations that have been observed on all shards, and
  // dispatch them right now. This will cover any unordered detachments on fields that all shards
  // just matched on; if a LogicalRegionField has been destroyed on all shards, that means its
  // detachment has also been emitted on all shards. This makes it safe to later block on any of
  // those detachments, since they are all guaranteed to be in the task stream now, and will
  // eventually complete.
  runtime_->progress_unordered_operations();
  // Put all the matched fields into the ordered queue, in the same order as the match result,
  // which is the same order that all shards will see.
  for (const auto& item : match.output()) {
    auto it = infos.find(item);
    assert(it != infos.end());
    ordered_free_fields_.push_back(std::move(it->second));
    infos.erase(it);
  }
  // All fields that weren't matched can go back into the unordered queue, to be included in the
  // next consensus match that we run.
  for (auto& [_, info] : infos) {
    unordered_free_fields_.push_back(std::move(info));
  }
  matches_.pop_front();
  info_for_match_items_.pop_front();
}

}  // namespace legate::detail
