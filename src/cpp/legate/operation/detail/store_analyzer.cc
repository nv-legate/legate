/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/store_analyzer.h>

#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/legion_utilities.h>
#include <legate/utilities/detail/traced_exception.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace legate::detail {

////////////////
// ProjectionSet
////////////////

void ProjectionSet::insert(Legion::PrivilegeMode new_privilege,
                           const StoreProjection& store_proj,
                           bool relax_interference_checks)
{
  // Pruning out the streaming discard mask makes the privilege coalescing code much
  // simpler. Otherwise, we would need to handle double the amount of combinations, because now
  // every combination of read, write, read-write also potentially has the output discard mask
  // applied.
  if (has_privilege(new_privilege, LEGION_DISCARD_OUTPUT_MASK)) {
    had_streaming_discard_ = true;
    new_privilege          = ignore_privilege(new_privilege, LEGION_DISCARD_OUTPUT_MASK);
  }
  if (store_projs_.empty()) {
    privilege_ = new_privilege;
  } else if (privilege_ != new_privilege && privilege_ != LEGION_NO_ACCESS &&
             new_privilege != LEGION_NO_ACCESS) {
    // conflicting privileges are promoted to a single read-write privilege
    privilege_ = LEGION_READ_WRITE;
  }
  store_projs_.emplace(store_proj);
  is_key_ = is_key_ || store_proj.is_key;

  if (privilege_ != LEGION_READ_ONLY && privilege_ != LEGION_NO_ACCESS && store_projs_.size() > 1 &&
      !relax_interference_checks) {
    throw TracedException<InterferingStoreError>{};
  }
}

// ==========================================================================================

void FieldSet::insert(Legion::FieldID field_id,
                      Legion::PrivilegeMode privilege,
                      const StoreProjection& store_proj,
                      bool relax_interference_checks)
{
  field_projs_[field_id].insert(privilege, store_proj, relax_interference_checks);
}

std::uint32_t FieldSet::num_requirements() const
{
  return static_cast<std::uint32_t>(coalesced_.size());
}

std::uint32_t FieldSet::get_requirement_index(Legion::PrivilegeMode privilege,
                                              const StoreProjection& store_proj,
                                              Legion::FieldID field_id) const
{
  for (auto priv : {privilege, LEGION_READ_WRITE, LEGION_READ_WRITE | LEGION_DISCARD_OUTPUT_MASK}) {
    if (const auto it = req_indices_.find({{priv, store_proj}, field_id});
        it != req_indices_.end()) {
      return it->second;
    }
  }
  LEGATE_ABORT("Failed to find privilege ", privilege, " in coalesced requirement indices");
}

void FieldSet::coalesce()
{
  std::size_t num_fields = 0;

  for (const auto& [field_id, proj_set] : field_projs_) {
    auto priv                  = proj_set.privilege();
    const auto proj_set_is_key = proj_set.is_key();
    const auto had_discard     = proj_set.had_streaming_discard();

    // We need to coalesce fields with DISCARD flag as a separate Legion Region Requirement
    if (had_discard) {
      priv |= LEGION_DISCARD_OUTPUT_MASK;
    }

    for (const auto& store_proj : proj_set.store_projs()) {
      auto& [fields, is_key, has_streaming_discard] = coalesced_[{priv, store_proj}];

      fields.emplace_back(field_id);
      if (proj_set_is_key) {
        is_key = true;
      }
      if (had_discard) {
        has_streaming_discard = true;
      }
      ++num_fields;
    }
  }

  // This will over-reserve
  req_indices_.reserve(num_fields);
  for (auto&& [idx, values] : enumerate(coalesced_)) {
    auto&& [key, entry] = values;

    for (const auto& field : entry.fields) {
      req_indices_[{key, field}] = static_cast<std::uint32_t>(idx);
    }
  }
}

namespace {

template <typename Launcher>
constexpr bool is_single_v = false;
template <>
constexpr bool is_single_v<Legion::TaskLauncher> = true;
template <>
constexpr bool is_single_v<Legion::IndexTaskLauncher> = false;

}  // namespace

template <typename Launcher>
void FieldSet::populate_launcher(Launcher* task, const Legion::LogicalRegion& region) const
{
  task->region_requirements.reserve(coalesced_.size());
  for (auto&& [key, entry] : coalesced_) {
    auto&& [fields, is_key, has_streaming_discard] = entry;
    auto&& [privilege, store_proj]                 = key;
    auto req = store_proj.template create_requirement<is_single_v<Launcher>>(
      region,
      fields,
      has_streaming_discard ? privilege | LEGION_DISCARD_OUTPUT_MASK : privilege,
      is_key);

    if (has_privilege(privilege, LEGION_WRITE_ONLY) && has_streaming_discard) {
      req.add_flags(LEGION_SUPPRESS_WARNINGS_FLAG);
    }

    task->region_requirements.emplace_back(std::move(req));
  }
}

//////////////////////
// RequirementAnalyzer
//////////////////////

void RequirementAnalyzer::insert(const Legion::LogicalRegion& region,
                                 Legion::FieldID field_id,
                                 Legion::PrivilegeMode privilege,
                                 const StoreProjection& store_proj)
{
  field_sets_[region].first.insert(field_id, privilege, store_proj, relax_interference_checks_);
}

std::uint32_t RequirementAnalyzer::get_requirement_index(const Legion::LogicalRegion& region,
                                                         Legion::PrivilegeMode privilege,
                                                         const StoreProjection& store_proj,
                                                         Legion::FieldID field_id) const
{
  const auto finder = field_sets_.find(region);
  LEGATE_ASSERT(finder != field_sets_.end());
  const auto& [field_set, req_offset] = finder->second;
  return req_offset + field_set.get_requirement_index(privilege, store_proj, field_id);
}

void RequirementAnalyzer::analyze_requirements()
{
  std::uint32_t num_reqs = 0;
  for (auto&& [_, entry] : field_sets_) {
    auto& [field_set, req_offset] = entry;
    field_set.coalesce();
    req_offset = num_reqs;
    num_reqs += field_set.num_requirements();
  }
}

void RequirementAnalyzer::populate_launcher(Legion::IndexTaskLauncher& task) const
{
  populate_launcher_(&task);
}

void RequirementAnalyzer::populate_launcher(Legion::TaskLauncher& task) const
{
  populate_launcher_(&task);
}

template <typename Launcher>
void RequirementAnalyzer::populate_launcher_(Launcher* task) const
{
  for (auto&& [region, entry] : field_sets_) {
    entry.first.populate_launcher(task, region);
  }
}

////////////////////////////
// OutputRequirementAnalyzer
////////////////////////////

void OutputRequirementAnalyzer::insert(std::uint32_t dim,
                                       const Legion::FieldSpace& field_space,
                                       Legion::FieldID field_id)
{
  auto& req_info = req_infos_[field_space];
  // TODO(wonchanl): This should be checked when alignment constraints are set on unbound stores
  LEGATE_ASSERT(ReqInfo::UNSET == req_info.dim || req_info.dim == dim);
  req_info.dim = dim;
  field_groups_[field_space].insert(field_id);
}

std::uint32_t OutputRequirementAnalyzer::get_requirement_index(
  const Legion::FieldSpace& field_space, Legion::FieldID) const
{
  auto finder = req_infos_.find(field_space);
  LEGATE_ASSERT(finder != req_infos_.end());
  return finder->second.req_idx;
}

void OutputRequirementAnalyzer::analyze_requirements()
{
  for (const auto& [idx, rest] : legate::detail::enumerate(field_groups_)) {
    auto&& [field_space, _] = rest;

    req_infos_[field_space].req_idx = idx;
  }
}

void OutputRequirementAnalyzer::populate_output_requirements(
  std::vector<Legion::OutputRequirement>& out_reqs) const
{
  out_reqs.reserve(field_groups_.size());
  for (auto&& [field_space, fields] : field_groups_) {
    auto&& [dim, _] = req_infos_.at(field_space);

    out_reqs.emplace_back(field_space, fields, dim, true /*global_indexing*/);
  }
}

/////////////////
// FutureAnalyzer
/////////////////

void FutureAnalyzer::insert(Legion::Future future) { futures_.emplace_back(std::move(future)); }

void FutureAnalyzer::insert(Legion::FutureMap future_map)
{
  future_maps_.emplace_back(std::move(future_map));
}

std::int32_t FutureAnalyzer::get_index(const Legion::Future& future) const
{
  return future_indices_.at(future);
}

std::int32_t FutureAnalyzer::get_index(const Legion::FutureMap& future_map) const
{
  return future_map_indices_.at(future_map);
}

void FutureAnalyzer::analyze_futures()
{
  std::int32_t index = 0;
  for (auto&& future : futures_) {
    if (const auto [it, inserted] = future_indices_.try_emplace(future); inserted) {
      it->second = index++;
      coalesced_futures_.emplace_back(future);
    }
  }
  for (auto&& future_map : future_maps_) {
    if (const auto [it, inserted] = future_map_indices_.try_emplace(future_map); inserted) {
      it->second = index++;
      coalesced_future_maps_.emplace_back(future_map);
    }
  }
}

namespace {

template <typename T, typename U>
void append_to_vec(std::vector<T>* dest, const U& span)
{
  dest->insert(dest->end(), span.begin(), span.end());
}

}  // namespace

void FutureAnalyzer::populate_launcher(Legion::IndexTaskLauncher& task) const
{
  append_to_vec(&task.futures, coalesced_futures_);
  append_to_vec(&task.point_futures, coalesced_future_maps_);
}

void FutureAnalyzer::populate_launcher(Legion::TaskLauncher& task) const
{
  LEGATE_ASSERT(coalesced_future_maps_.empty());
  append_to_vec(&task.futures, coalesced_futures_);
}

// ==========================================================================================

void StoreAnalyzer::insert(const InternalSharedPtr<LogicalRegionField>& region_field,
                           Legion::PrivilegeMode privilege,
                           const StoreProjection& store_proj)
{
  req_analyzer_.insert(region_field->region(), region_field->field_id(), privilege, store_proj);
}

void StoreAnalyzer::insert(std::uint32_t dim,
                           const Legion::FieldSpace& field_space,
                           Legion::FieldID field_id)
{
  out_analyzer_.insert(dim, field_space, field_id);
}

void StoreAnalyzer::insert(Legion::Future future) { fut_analyzer_.insert(std::move(future)); }

void StoreAnalyzer::insert(Legion::FutureMap future_map)
{
  fut_analyzer_.insert(std::move(future_map));
}

void StoreAnalyzer::analyze()
{
  req_analyzer_.analyze_requirements();
  out_analyzer_.analyze_requirements();
  fut_analyzer_.analyze_futures();
}

std::uint32_t StoreAnalyzer::get_index(const Legion::LogicalRegion& region,
                                       Legion::PrivilegeMode privilege,
                                       const StoreProjection& store_proj,
                                       Legion::FieldID field_id) const
{
  return req_analyzer_.get_requirement_index(region, privilege, store_proj, field_id);
}

std::uint32_t StoreAnalyzer::get_index(const Legion::FieldSpace& field_space,
                                       Legion::FieldID field_id) const
{
  return out_analyzer_.get_requirement_index(field_space, field_id);
}

std::int32_t StoreAnalyzer::get_index(const Legion::Future& future) const
{
  return fut_analyzer_.get_index(future);
}

std::int32_t StoreAnalyzer::get_index(const Legion::FutureMap& future_map) const
{
  return fut_analyzer_.get_index(future_map);
}

}  // namespace legate::detail
