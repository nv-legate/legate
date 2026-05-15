/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/store_projection.h>

namespace legate::detail {

inline StoreProjection::StoreProjection(Legion::LogicalPartition _partition,
                                        Legion::ProjectionID _proj_id)
  : partition{std::move(_partition)}, proj_id{_proj_id}
{
}

inline bool StoreProjection::operator<(const StoreProjection& other) const
{
  return std::tie(partition, proj_id, redop) <
         std::tie(other.partition, other.proj_id, other.redop);
}

inline bool StoreProjection::operator==(const StoreProjection& other) const
{
  return partition == other.partition && proj_id == other.proj_id && redop == other.redop;
}

inline void StoreProjection::set_reduction_op(GlobalRedopID _redop) { redop = _redop; }

inline std::size_t StoreProjection::hash() const noexcept
{
  return hash_all(partition, proj_id, redop);
}

}  // namespace legate::detail
