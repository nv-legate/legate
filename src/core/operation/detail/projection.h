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

#include "core/utilities/detail/hash.h"
#include "core/utilities/typedefs.h"

namespace legate::detail {

struct BaseProjectionInfo {
  BaseProjectionInfo() = default;

  BaseProjectionInfo(Legion::LogicalPartition partition, Legion::ProjectionID proj_id);

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

  void set_reduction_op(Legion::ReductionOpID _redop);

  [[nodiscard]] size_t hash() const noexcept;

  Legion::LogicalPartition partition{Legion::LogicalPartition::NO_PART};
  Legion::ProjectionID proj_id{};
  Legion::ReductionOpID redop{-1};
};

struct ProjectionInfo : BaseProjectionInfo {
  using BaseProjectionInfo::BaseProjectionInfo;
  using BaseProjectionInfo::populate_requirement;

  template <bool SINGLE>
  void populate_requirement(Legion::RegionRequirement& requirement,
                            const Legion::LogicalRegion& region,
                            const std::vector<Legion::FieldID>& fields,
                            Legion::PrivilegeMode privilege) const;

  bool is_key{};
};

}  // namespace legate::detail

#include "core/operation/detail/projection.inl"
