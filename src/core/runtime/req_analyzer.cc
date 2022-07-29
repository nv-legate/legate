/* Copyright 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "core/runtime/req_analyzer.h"
#include "core/runtime/launcher.h"

namespace legate {

/////////////
// Projection
/////////////

Projection::Projection(Legion::LogicalPartition p, Legion::ProjectionID pr)
  : partition(p), proj_id(pr)
{
}

void Projection::set_reduction_op(Legion::ReductionOpID r) { redop = r; }

/////////////////
// ProjectionInfo
/////////////////

ProjectionInfo::ProjectionInfo(const Projection* proj,
                               Legion::MappingTagID _tag,
                               Legion::RegionFlags _flags)
  : partition(proj->partition), proj_id(proj->proj_id), redop(proj->redop), tag(_tag), flags(_flags)
{
}

bool ProjectionInfo::operator<(const ProjectionInfo& other) const
{
  if (partition < other.partition)
    return true;
  else if (other.partition < partition)
    return false;
  if (proj_id < other.proj_id)
    return true;
  else if (proj_id > other.proj_id)
    return false;
  if (redop < other.redop)
    return true;
  else if (redop > other.redop)
    return false;
  if (tag < other.tag)
    return true;
  else
    return false;
}

bool ProjectionInfo::operator==(const ProjectionInfo& other) const
{
  return partition == other.partition && proj_id == other.proj_id && redop == other.redop &&
         tag == other.tag && flags == other.flags;
}

void ProjectionInfo::populate_launcher(Legion::TaskLauncher* task,
                                       const Legion::LogicalRegion& region,
                                       const std::vector<Legion::FieldID>& fields,
                                       Legion::PrivilegeMode privilege) const
{
  Legion::RegionRequirement legion_req;

  if (LEGION_REDUCE == privilege) {
#ifdef DEBUG_LEGATE
    assert(redop != -1);
#endif
    new (&legion_req) Legion::RegionRequirement(region, redop, LEGION_EXCLUSIVE, region, tag);
  } else {
    new (&legion_req) Legion::RegionRequirement(region, privilege, LEGION_EXCLUSIVE, region, tag);
  }

  legion_req.add_fields(fields).add_flags(flags);
  task->add_region_requirement(legion_req);
}

void ProjectionInfo::populate_launcher(Legion::IndexTaskLauncher* task,
                                       const Legion::LogicalRegion& region,
                                       const std::vector<Legion::FieldID>& fields,
                                       Legion::PrivilegeMode privilege) const
{
  Legion::RegionRequirement legion_req;

  // Broadcast
  if (Legion::LogicalPartition::NO_PART == partition) {
    if (LEGION_REDUCE == privilege) {
#ifdef DEBUG_LEGATE
      assert(redop != -1);
#endif
      new (&legion_req) Legion::RegionRequirement(region, redop, LEGION_EXCLUSIVE, region, tag);
    } else {
      new (&legion_req) Legion::RegionRequirement(region, privilege, LEGION_EXCLUSIVE, region, tag);
    }
  } else {
    if (LEGION_REDUCE == privilege) {
#ifdef DEBUG_LEGATE
      assert(redop != -1);
#endif
      new (&legion_req)
        Legion::RegionRequirement(partition, proj_id, redop, LEGION_EXCLUSIVE, region, tag);
    } else {
      new (&legion_req)
        Legion::RegionRequirement(partition, proj_id, privilege, LEGION_EXCLUSIVE, region, tag);
    }
  }

  legion_req.add_fields(fields).add_flags(flags);
  task->add_region_requirement(legion_req);
}

////////////////
// ProjectionSet
////////////////

void ProjectionSet::insert(Legion::PrivilegeMode new_privilege, const ProjectionInfo* proj_info)
{
  if (proj_infos.empty()) privilege = new_privilege;
  // conflicting privileges are promoted to a single read-write privilege
  else if (privilege != new_privilege)
    privilege = LEGION_READ_WRITE;
  proj_infos.emplace(*proj_info);

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
                      const ProjectionInfo* proj_info)
{
  field_projs_[field_id].insert(privilege, proj_info);
}

uint32_t FieldSet::num_requirements() const { return static_cast<uint32_t>(coalesced_.size()); }

uint32_t FieldSet::get_requirement_index(Legion::PrivilegeMode privilege,
                                         const ProjectionInfo* proj_info) const
{
  auto finder = req_indices_.find(Key(privilege, *proj_info));
#ifdef DEBUG_LEGATE
  assert(finder != req_indices_.end());
#endif
  return finder->second;
}

void FieldSet::coalesce()
{
  for (const auto& entry : field_projs_) {
    const auto& proj_set = entry.second;
    for (const auto& proj_info : proj_set.proj_infos)
      coalesced_[Key(proj_set.privilege, proj_info)].push_back(entry.first);
  }
  uint32_t idx = 0;
  for (const auto& entry : coalesced_) req_indices_[entry.first] = idx++;
}

void FieldSet::populate_launcher(Legion::IndexTaskLauncher* task,
                                 const Legion::LogicalRegion& region) const
{
  for (auto& entry : coalesced_) {
    auto privilege        = entry.first.first;
    const auto& proj_info = entry.first.second;
    const auto& fields    = entry.second;
    proj_info.populate_launcher(task, region, fields, privilege);
  }
}

void FieldSet::populate_launcher(Legion::TaskLauncher* task,
                                 const Legion::LogicalRegion& region) const
{
  for (auto& entry : coalesced_) {
    auto privilege        = entry.first.first;
    const auto& proj_info = entry.first.second;
    const auto& fields    = entry.second;
    proj_info.populate_launcher(task, region, fields, privilege);
  }
}

//////////////////////
// RequirementAnalyzer
//////////////////////

RequirementAnalyzer::~RequirementAnalyzer() {}

void RequirementAnalyzer::insert(const Legion::LogicalRegion& region,
                                 Legion::FieldID field_id,
                                 Legion::PrivilegeMode privilege,
                                 const ProjectionInfo* proj_info)
{
  field_sets_[region].first.insert(field_id, privilege, proj_info);
}

uint32_t RequirementAnalyzer::get_requirement_index(const Legion::LogicalRegion& region,
                                                    Legion::PrivilegeMode privilege,
                                                    const ProjectionInfo* proj_info) const
{
  auto finder = field_sets_.find(region);
#ifdef DEBUG_LEGATE
  assert(finder != field_sets_.end());
#endif
  auto& field_set  = finder->second.first;
  auto& req_offset = finder->second.second;
  return req_offset + field_set.get_requirement_index(privilege, proj_info);
}

void RequirementAnalyzer::analyze_requirements()
{
  uint32_t num_reqs = 0;
  for (auto& entry : field_sets_) {
    auto& field_set  = entry.second.first;
    auto& req_offset = entry.second.second;

    field_set.coalesce();
    req_offset = num_reqs;
    num_reqs += field_set.num_requirements();
  }
}

void RequirementAnalyzer::populate_launcher(Legion::IndexTaskLauncher* task) const
{
  for (auto& entry : field_sets_) {
    const auto& field_set = entry.second.first;
    field_set.populate_launcher(task, entry.first);
  }
}

void RequirementAnalyzer::populate_launcher(Legion::TaskLauncher* task) const
{
  for (auto& entry : field_sets_) {
    const auto& field_set = entry.second.first;
    field_set.populate_launcher(task, entry.first);
  }
}

}  // namespace legate
