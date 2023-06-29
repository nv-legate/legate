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

#pragma once

#include <tuple>
#include "core/utilities/typedefs.h"

namespace legate::detail {

struct ProjectionInfo {
  ProjectionInfo() {}
  ProjectionInfo(Legion::LogicalPartition partition, Legion::ProjectionID proj_id);

  ProjectionInfo(const ProjectionInfo&)            = default;
  ProjectionInfo& operator=(const ProjectionInfo&) = default;

  bool operator<(const ProjectionInfo& other) const;
  bool operator==(const ProjectionInfo& other) const;

  // TODO: Ideally we want this method to return a requirement, instead of taking an inout argument.
  // We go with an inout parameter for now, as RegionRequirement doesn't have a move
  // constructor/assignment.
  template <bool SINGLE>
  void populate_requirement(Legion::RegionRequirement& requirement,
                            const Legion::LogicalRegion& region,
                            const std::vector<Legion::FieldID>& fields,
                            Legion::PrivilegeMode privilege) const;

  void set_reduction_op(Legion::ReductionOpID _redop) { redop = _redop; }

  Legion::LogicalPartition partition{Legion::LogicalPartition::NO_PART};
  Legion::ProjectionID proj_id{0};
  Legion::ReductionOpID redop{-1};
  Legion::MappingTagID tag{0};
  Legion::RegionFlags flags{LEGION_NO_FLAG};
};

}  // namespace legate::detail
