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
  : runtime_(runtime), shape_(shape), field_size_(field_size)
{
  log_legate.debug() << "Field manager " << this << " created for shape " << shape
                     << ", field_size " << field_size;
}

std::shared_ptr<LogicalRegionField> FieldManager::allocate_field()
{
  issue_field_match();
  while (!ordered_free_fields_.empty() || !matches_.empty()) {
    // If there's a field that every shard is guaranteed to have, re-use that.
    if (!ordered_free_fields_.empty()) {
      const auto& field = ordered_free_fields_.front();
      auto* rf          = new LogicalRegionField(this, field.first, field.second);
      log_legate.debug("Field %u recycled in field manager %p", field.second, this);
      ordered_free_fields_.pop_front();
      return std::shared_ptr<LogicalRegionField>(rf);
    }
    // If there are any field matches we haven't processed yet, process the next one, then go back
    // and check if any fields were just added to the "ordered" queue.
    process_next_field_match();
  }
  // If there are no more field matches to process, then we completely failed to reuse a field.
  auto rgn_mgr   = runtime_->find_or_create_region_manager(shape_);
  auto [lr, fid] = rgn_mgr->allocate_field(field_size_);
  auto* rf       = new LogicalRegionField(this, lr, fid);
  log_legate.debug("Field %u created in field manager %p", fid, this);
  return std::shared_ptr<LogicalRegionField>(rf);
}

std::shared_ptr<LogicalRegionField> FieldManager::import_field(const Legion::LogicalRegion& region,
                                                               Legion::FieldID field_id)
{
  // Import the region only if the region manager is created fresh
  auto rgn_mgr = runtime_->find_or_create_region_manager(shape_);
  if (!rgn_mgr->has_space()) rgn_mgr->import_region(region);
  log_legate.debug("Field %u imported in field manager %p", field_id, this);
  return std::make_shared<LogicalRegionField>(this, region, field_id);
}

void FieldManager::free_field(const Legion::LogicalRegion& region,
                              Legion::FieldID field_id,
                              bool unordered)
{
  if (unordered && runtime_->consensus_match_required()) {
    log_legate.debug("Field %u freed locally in field manager %p", field_id, this);
    unordered_free_fields_.push_back(FreeField(region, field_id));
  } else {
    log_legate.debug("Field %u freed in-order in field manager %p", field_id, this);
    ordered_free_fields_.push_back(FreeField(region, field_id));
  }
}

void FieldManager::issue_field_match()
{
  // Check if there are any freed fields that are shared across all the shards. We have to
  // test this deterministically no matter what, even if we don't have any fields to offer
  // ourselves, because this is a collective with other shards.
  if (++field_match_counter_ < runtime_->field_reuse_freq()) return;
  field_match_counter_ = 0;
  // We need to separately record the region that corresponds to each item in this match, because
  // the match itself only uses a subset of the full region info.
  auto& regions = regions_for_match_items_.emplace_back();
  std::vector<MatchItem> input;
  input.reserve(unordered_free_fields_.size());
  for (const auto& field : unordered_free_fields_) {
    MatchItem item{field.first.get_tree_id(), field.second};
    input.push_back(item);
    regions[item] = field.first;
  }
  assert(regions.size() == unordered_free_fields_.size());
  unordered_free_fields_.clear();
  // Dispatch the match and put it on the queue of outstanding matches, but don't block on it yet.
  // We'll do that when we run out of ordered fields.
  matches_.push_back(runtime_->issue_consensus_match(std::move(input)));
  log_legate.debug(
    "Consensus match emitted with %zu local fields in field manager %p", regions.size(), this);
}

void FieldManager::process_next_field_match()
{
  assert(!matches_.empty());
  auto& match   = matches_.front();
  auto& regions = regions_for_match_items_.front();
  match.wait();
  log_legate.debug("Consensus match result in field manager %p: %zu/%zu fields matched",
                   this,
                   match.output().size(),
                   match.input().size());
  // Put all the matched fields into the ordered queue, in the same order as the match result,
  // which is the same order that all shards will see.
  for (const auto& item : match.output()) {
    auto it = regions.find(item);
    assert(it != regions.end());
    ordered_free_fields_.push_back(FreeField(it->second, item.fid));
    regions.erase(it);
  }
  // All fields that weren't matched can go back into the unordered queue, to be included in the
  // next consensus match that we run.
  for (const auto& [item, lr] : regions) {
    unordered_free_fields_.push_back(FreeField(lr, item.fid));
  }
  matches_.pop_front();
  regions_for_match_items_.pop_front();
}

}  // namespace legate::detail
