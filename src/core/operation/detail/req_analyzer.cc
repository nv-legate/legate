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

#include "core/operation/detail/req_analyzer.h"
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

////////////////
// ProjectionSet
////////////////

void ProjectionSet::insert(Legion::PrivilegeMode new_privilege, const ProjectionInfo& proj_info)
{
  if (proj_infos.empty()) privilege = new_privilege;
  // conflicting privileges are promoted to a single read-write privilege
  else if (privilege != new_privilege)
    privilege = LEGION_READ_WRITE;
  proj_infos.emplace(proj_info);

  if (privilege != LEGION_READ_ONLY && proj_infos.size() > 1) {
    log_legate.error("Interfering requirements are found");
    LEGATE_ABORT;
  }
}

///////////
// FieldSet
///////////

void FieldSet::insert(Legion::FieldID field_id,
                      Legion::PrivilegeMode privilege,
                      const ProjectionInfo& proj_info)
{
  field_projs_[field_id].insert(privilege, proj_info);
}

uint32_t FieldSet::num_requirements() const { return static_cast<uint32_t>(coalesced_.size()); }

uint32_t FieldSet::get_requirement_index(Legion::PrivilegeMode privilege,
                                         const ProjectionInfo& proj_info) const
{
  auto finder = req_indices_.find(Key(privilege, proj_info));
  if (req_indices_.end() == finder) finder = req_indices_.find(Key(LEGION_READ_WRITE, proj_info));
#ifdef DEBUG_LEGATE
  assert(finder != req_indices_.end());
#endif
  return finder->second;
}

void FieldSet::coalesce()
{
  for (const auto& [field_id, proj_set] : field_projs_) {
    for (const auto& proj_info : proj_set.proj_infos)
      coalesced_[Key(proj_set.privilege, proj_info)].push_back(field_id);
  }
  uint32_t idx = 0;
  for (const auto& [key, _] : coalesced_) req_indices_[key] = idx++;
}

namespace {

template <class Launcher>
constexpr bool is_single = false;
template <>
constexpr bool is_single<Legion::TaskLauncher> = true;
template <>
constexpr bool is_single<Legion::IndexTaskLauncher> = false;

}  // namespace

template <class Launcher>
void FieldSet::populate_launcher(Launcher* task, const Legion::LogicalRegion& region) const
{
  for (auto& [key, fields] : coalesced_) {
    auto& [privilege, proj_info] = key;
    task->region_requirements.push_back(Legion::RegionRequirement());
    auto& requirement = task->region_requirements.back();
    proj_info.template populate_requirement<is_single<Launcher>>(
      requirement, region, fields, privilege);
  }
}

//////////////////////
// RequirementAnalyzer
//////////////////////

RequirementAnalyzer::~RequirementAnalyzer() {}

void RequirementAnalyzer::insert(const Legion::LogicalRegion& region,
                                 Legion::FieldID field_id,
                                 Legion::PrivilegeMode privilege,
                                 const ProjectionInfo& proj_info)
{
  field_sets_[region].first.insert(field_id, privilege, proj_info);
}

uint32_t RequirementAnalyzer::get_requirement_index(const Legion::LogicalRegion& region,
                                                    Legion::PrivilegeMode privilege,
                                                    const ProjectionInfo& proj_info) const
{
  auto finder = field_sets_.find(region);
#ifdef DEBUG_LEGATE
  assert(finder != field_sets_.end());
#endif
  auto& [field_set, req_offset] = finder->second;
  return req_offset + field_set.get_requirement_index(privilege, proj_info);
}

void RequirementAnalyzer::analyze_requirements()
{
  uint32_t num_reqs = 0;
  for (auto& [_, entry] : field_sets_) {
    auto& [field_set, req_offset] = entry;
    field_set.coalesce();
    req_offset = num_reqs;
    num_reqs += field_set.num_requirements();
  }
}

void RequirementAnalyzer::populate_launcher(Legion::IndexTaskLauncher* task) const
{
  _populate_launcher(task);
}

void RequirementAnalyzer::populate_launcher(Legion::TaskLauncher* task) const
{
  _populate_launcher(task);
}

template <class Launcher>
void RequirementAnalyzer::_populate_launcher(Launcher* task) const
{
  for (auto& [region, entry] : field_sets_) entry.first.populate_launcher(task, region);
}

////////////////////////////
// OutputRequirementAnalyzer
////////////////////////////

OutputRequirementAnalyzer::~OutputRequirementAnalyzer() {}

void OutputRequirementAnalyzer::insert(int32_t dim,
                                       const Legion::FieldSpace& field_space,
                                       Legion::FieldID field_id)
{
  auto& req_info = req_infos_[field_space];
#ifdef DEBUG_LEGATE
  // TODO: This should be checked when alignment constraints are set on unbound stores
  assert(-1 == req_info.dim || req_info.dim == dim);
#endif
  req_info.dim = dim;
  field_groups_[field_space].insert(field_id);
}

uint32_t OutputRequirementAnalyzer::get_requirement_index(const Legion::FieldSpace& field_space,
                                                          Legion::FieldID field_id) const
{
  auto finder = req_infos_.find(field_space);
#ifdef DEBUG_LEGATE
  assert(finder != req_infos_.end());
#endif
  return finder->second.req_idx;
}

void OutputRequirementAnalyzer::analyze_requirements()
{
  uint32_t idx = 0;
  for (const auto& [field_space, _] : field_groups_) req_infos_[field_space].req_idx = idx++;
}

void OutputRequirementAnalyzer::populate_output_requirements(
  std::vector<Legion::OutputRequirement>& out_reqs) const
{
  for (auto& [field_space, fields] : field_groups_) {
    auto& [dim, _] = req_infos_.at(field_space);
    out_reqs.emplace_back(field_space, fields, dim, true /*global_indexing*/);
  }
}

}  // namespace legate::detail
