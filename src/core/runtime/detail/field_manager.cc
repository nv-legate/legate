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

FieldManager::FieldManager(InternalSharedPtr<Shape> shape, std::uint32_t field_size)
  : shape_{std::move(shape)}, field_size_{field_size}
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Field manager " << this << " created for shape " << shape_->to_string()
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

void FieldManager::free_field(FreeFieldInfo free_field_info, bool /*unordered*/)
{
  log_legate().debug("Field %u freed in-order in field manager %p",
                     free_field_info.field_id,
                     static_cast<void*>(this));
  auto& info = ordered_free_fields_.emplace_back(std::move(free_field_info));
  Runtime::get_runtime()->discard_field(info.region, info.field_id);
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
  auto rgn_mgr   = Runtime::get_runtime()->find_or_create_region_manager(shape_->index_space());
  auto [lr, fid] = rgn_mgr->allocate_field(field_size_);
  auto* rf       = new LogicalRegionField{this, lr, fid};

  log_legate().debug("Field %u created in field manager %p", fid, static_cast<void*>(this));
  return InternalSharedPtr<LogicalRegionField>{rf};
}

// ==========================================================================================

ConsensusMatchingFieldManager::ConsensusMatchingFieldManager(InternalSharedPtr<Shape> shape,
                                                             std::uint32_t field_size)
  : FieldManager{std::move(shape), field_size}
{
  if (shape_->ready()) {
    calculate_match_credit(shape_.get());
  } else {
    auto* runtime       = Runtime::get_runtime();
    field_match_credit_ = runtime->field_reuse_freq();
    runtime->find_or_create_region_manager(shape_->index_space())
      ->record_pending_match_credit_update(this);
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

void ConsensusMatchingFieldManager::free_field(FreeFieldInfo free_field_info, bool unordered)
{
  if (unordered) {
    log_legate().debug("Field %u freed locally in field manager %p",
                       free_field_info.field_id,
                       static_cast<void*>(this));
    unordered_free_fields_.emplace_back(std::move(free_field_info));
  } else {
    FieldManager::free_field(std::move(free_field_info), unordered);
  }
}

void ConsensusMatchingFieldManager::calculate_match_credit(const Shape* initiator)
{
  if (!shape_->ready()) {
    LegateAssert(initiator->ready());
    shape_->copy_extents_from(*initiator);
  }
  const auto size = shape_->volume() * field_size_;
  if (size > Config::max_field_reuse_size) {
    LegateCheck(Config::max_field_reuse_size > 0);
    field_match_credit_ = (size + Config::max_field_reuse_size - 1) / Config::max_field_reuse_size;
  }
}

void ConsensusMatchingFieldManager::issue_field_match()
{
  auto* runtime = Runtime::get_runtime();
  // Check if there are any freed fields that are shared across all the shards. We have to
  // test this deterministically no matter what, even if we don't have any fields to offer
  // ourselves, because this is a collective with other shards.
  field_match_counter_ += field_match_credit_;
  if (field_match_counter_ < runtime->field_reuse_freq()) {
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
  LegateCheck(infos.size() == unordered_free_fields_.size());
  unordered_free_fields_.clear();
  // Dispatch the match and put it on the queue of outstanding matches, but don't block on it yet.
  // We'll do that when we run out of ordered fields.
  matches_.push_back(runtime->issue_consensus_match(std::move(input)));
  log_legate().debug("Consensus match emitted with %zu local fields in field manager %p",
                     infos.size(),
                     static_cast<void*>(this));
}

void ConsensusMatchingFieldManager::process_next_field_match()
{
  LegateCheck(!matches_.empty());
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
  Runtime::get_runtime()->progress_unordered_operations();
  // Put all the matched fields into the ordered queue, in the same order as the match result,
  // which is the same order that all shards will see.
  for (const auto& item : match.output()) {
    auto it = infos.find(item);
    LegateCheck(it != infos.end());
    FieldManager::free_field(std::move(it->second), false /*unordered*/);
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
