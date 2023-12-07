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

#include "core/operation/detail/projection.h"

namespace legate::detail {

inline BaseProjectionInfo::BaseProjectionInfo(Legion::LogicalPartition _partition,
                                              Legion::ProjectionID _proj_id)
  : partition{_partition}, proj_id{_proj_id}
{
}

inline bool BaseProjectionInfo::operator<(const BaseProjectionInfo& other) const
{
  return std::tie(partition, proj_id, redop) <
         std::tie(other.partition, other.proj_id, other.redop);
}

inline bool BaseProjectionInfo::operator==(const BaseProjectionInfo& other) const
{
  return partition == other.partition && proj_id == other.proj_id && redop == other.redop;
}

inline void BaseProjectionInfo::set_reduction_op(Legion::ReductionOpID _redop) { redop = _redop; }

inline size_t BaseProjectionInfo::hash() const noexcept
{
  return hash_all(partition, proj_id, redop);
}

// ==========================================================================================

template <bool SINGLE>
void ProjectionInfo::populate_requirement(Legion::RegionRequirement& requirement,
                                          const Legion::LogicalRegion& region,
                                          const std::vector<Legion::FieldID>& fields,
                                          Legion::PrivilegeMode privilege) const
{
  return populate_requirement<SINGLE>(requirement, region, fields, privilege, is_key);
}

}  // namespace legate::detail
