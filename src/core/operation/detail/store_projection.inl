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

#include "core/operation/detail/store_projection.h"

namespace legate::detail {

inline BaseStoreProjection::BaseStoreProjection(Legion::LogicalPartition _partition,
                                                Legion::ProjectionID _proj_id)
  : partition{std::move(_partition)}, proj_id{_proj_id}
{
}

inline bool BaseStoreProjection::operator<(const BaseStoreProjection& other) const
{
  return std::tie(partition, proj_id, redop) <
         std::tie(other.partition, other.proj_id, other.redop);
}

inline bool BaseStoreProjection::operator==(const BaseStoreProjection& other) const
{
  return partition == other.partition && proj_id == other.proj_id && redop == other.redop;
}

inline void BaseStoreProjection::set_reduction_op(Legion::ReductionOpID _redop) { redop = _redop; }

inline std::size_t BaseStoreProjection::hash() const noexcept
{
  return hash_all(partition, proj_id, redop);
}

// ==========================================================================================

template <bool SINGLE>
void StoreProjection::populate_requirement(Legion::RegionRequirement& requirement,
                                           const Legion::LogicalRegion& region,
                                           const std::vector<Legion::FieldID>& fields,
                                           Legion::PrivilegeMode privilege) const
{
  return populate_requirement<SINGLE>(requirement, region, fields, privilege, is_key);
}

}  // namespace legate::detail
