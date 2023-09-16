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

#pragma once

#include <tuple>
#include "core/utilities/typedefs.h"

namespace legate::detail {

struct BaseProjectionInfo {
  BaseProjectionInfo() = default;
  BaseProjectionInfo(Legion::LogicalPartition partition, Legion::ProjectionID proj_id);

  BaseProjectionInfo(const BaseProjectionInfo&)            = default;
  BaseProjectionInfo& operator=(const BaseProjectionInfo&) = default;

  bool operator<(const BaseProjectionInfo& other) const;
  bool operator==(const BaseProjectionInfo& other) const;

  // TODO: Ideally we want this method to return a requirement, instead of taking an inout argument.
  // We go with an inout parameter for now, as RegionRequirement doesn't have a move
  // constructor/assignment.
  template <bool SINGLE>
  void populate_requirement(Legion::RegionRequirement& requirement,
                            const Legion::LogicalRegion& region,
                            const std::vector<Legion::FieldID>& fields,
                            Legion::PrivilegeMode privilege,
                            bool is_key) const;

  void set_reduction_op(Legion::ReductionOpID _redop) { redop = _redop; }

  Legion::LogicalPartition partition{Legion::LogicalPartition::NO_PART};
  Legion::ProjectionID proj_id{0};
  Legion::ReductionOpID redop{-1};
};

struct ProjectionInfo : public BaseProjectionInfo {
  ProjectionInfo() = default;
  ProjectionInfo(Legion::LogicalPartition partition, Legion::ProjectionID proj_id);

  ProjectionInfo(const ProjectionInfo&)            = default;
  ProjectionInfo& operator=(const ProjectionInfo&) = default;

  using BaseProjectionInfo::populate_requirement;
  template <bool SINGLE>
  void populate_requirement(Legion::RegionRequirement& requirement,
                            const Legion::LogicalRegion& region,
                            const std::vector<Legion::FieldID>& fields,
                            Legion::PrivilegeMode privilege) const;

  bool is_key{false};
};

}  // namespace legate::detail
