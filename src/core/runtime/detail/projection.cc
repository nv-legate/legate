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

#include "core/runtime/detail/projection.h"
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

template <>
void ProjectionInfo::populate_requirement<true>(Legion::RegionRequirement& requirement,
                                                const Legion::LogicalRegion& region,
                                                const std::vector<Legion::FieldID>& fields,
                                                Legion::PrivilegeMode privilege) const
{
  auto parent = Runtime::get_runtime()->find_parent_region(region);

  if (LEGION_REDUCE == privilege) {
    new (&requirement) Legion::RegionRequirement(region, redop, LEGION_EXCLUSIVE, parent, tag);
  } else {
    new (&requirement) Legion::RegionRequirement(region, privilege, LEGION_EXCLUSIVE, parent, tag);
  }
  requirement.add_fields(fields).add_flags(flags);
}

template <>
void ProjectionInfo::populate_requirement<false>(Legion::RegionRequirement& requirement,
                                                 const Legion::LogicalRegion& region,
                                                 const std::vector<Legion::FieldID>& fields,
                                                 Legion::PrivilegeMode privilege) const
{
  if (Legion::LogicalPartition::NO_PART == partition) {
    populate_requirement<true>(requirement, region, fields, privilege);
    return;
  }

  auto parent = Runtime::get_runtime()->find_parent_region(region);

  if (LEGION_REDUCE == privilege) {
    new (&requirement)
      Legion::RegionRequirement(partition, proj_id, redop, LEGION_EXCLUSIVE, parent, tag);
  } else {
    new (&requirement)
      Legion::RegionRequirement(partition, proj_id, privilege, LEGION_EXCLUSIVE, parent, tag);
  }
  requirement.add_fields(fields).add_flags(flags);
}

ProjectionInfo::ProjectionInfo(Legion::LogicalPartition _partition, Legion::ProjectionID _proj_id)
  : partition(_partition), proj_id(_proj_id)
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

}  // namespace legate::detail
