/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/store_projection.h>

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

inline void BaseStoreProjection::set_reduction_op(GlobalRedopID _redop) { redop = _redop; }

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
