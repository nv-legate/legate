/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate/operation/detail/req_analyzer.h"

#include "legate/utilities/detail/enumerate.h"
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

namespace legate::detail {

////////////////
// ProjectionSet
////////////////

void ProjectionSet::insert(Legion::PrivilegeMode new_privilege,
                           const StoreProjection& store_proj,
                           bool relax_interference_checks)
{
  if (store_projs.empty()) {
    privilege = new_privilege;
    // conflicting privileges are promoted to a single read-write privilege
  } else if (privilege != new_privilege && privilege != NO_ACCESS && new_privilege != NO_ACCESS) {
    privilege = LEGION_READ_WRITE;
  }
  store_projs.emplace(store_proj);
  is_key = is_key || store_proj.is_key;

  if (privilege != LEGION_READ_ONLY && privilege != NO_ACCESS && store_projs.size() > 1 &&
      !relax_interference_checks) {
    throw TracedException<InterferingStoreError>{};
  }
}

///////////
// FieldSet
///////////

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
  auto finder = req_indices_.find({{privilege, store_proj}, field_id});
  if (req_indices_.end() == finder) {
    finder = req_indices_.find({{LEGION_READ_WRITE, store_proj}, field_id});
  }
  LEGATE_ASSERT(finder != req_indices_.end());
  return finder->second;
}

void FieldSet::coalesce()
{
  for (const auto& [field_id, proj_set] : field_projs_) {
    for (const auto& store_proj : proj_set.store_projs) {
      auto& [fields, is_key] = coalesced_[{proj_set.privilege, store_proj}];
      fields.emplace_back(field_id);
      is_key = is_key || proj_set.is_key;
    }
  }
  std::uint32_t idx = 0;
  for (const auto& [key, entry] : coalesced_) {
    for (const auto& field : entry.fields) {
      req_indices_[{key, field}] = idx;
    }
    ++idx;
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
    auto&& [fields, is_key]        = entry;
    auto&& [privilege, store_proj] = key;
    auto&& requirement = task->region_requirements.emplace_back(Legion::RegionRequirement{});

    store_proj.template populate_requirement<is_single_v<Launcher>>(
      requirement, region, fields, privilege, is_key);
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

template <typename T, typename U = T>
void append_to_vec(std::vector<T>* dest, const std::vector<U>& src)
{
  dest->insert(dest->end(), src.begin(), src.end());
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
