/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "legate/mapping/detail/instance_manager.h"

#include <utility>

namespace legate::mapping::detail {

// Silence pass-by-value since Legion::Domain is POD, and the move ctor just does the copy
// anyways. Unfortunately there is no way to check this programatically (e.g. via a
// static_assert).
inline RegionGroup::RegionGroup(std::set<Legion::LogicalRegion> regions_,
                                const Domain& bounding_box_  // NOLINT(modernize-pass-by-value)
                                )
  : regions{std::move(regions_)}, bounding_box{bounding_box_}
{
}

// ==========================================================================================

inline InstanceSet::InstanceSpec::InstanceSpec(Legion::Mapping::PhysicalInstance inst,
                                               InstanceMappingPolicy po)
  : instance{std::move(inst)}, policy{std::move(po)}
{
}

inline bool InstanceSet::empty() const { return instances_.empty() && pending_instances_.empty(); }

// ==========================================================================================

inline ReductionInstanceSet::ReductionInstanceSpec::ReductionInstanceSpec(
  GlobalRedopID op, Legion::Mapping::PhysicalInstance inst, InstanceMappingPolicy po)
  : redop{op}, instance{std::move(inst)}, policy{std::move(po)}
{
}

inline bool ReductionInstanceSet::empty() const { return instances_.empty(); }

// ==========================================================================================

inline BaseInstanceManager::FieldMemInfo::FieldMemInfo(Legion::RegionTreeID t,
                                                       Legion::FieldID f,
                                                       Memory m)
  : tid{t}, fid{f}, memory{m}
{
}

inline bool BaseInstanceManager::FieldMemInfo::operator==(const FieldMemInfo& rhs) const
{
  return tid == rhs.tid && fid == rhs.fid && memory == rhs.memory;
}

inline std::size_t BaseInstanceManager::FieldMemInfo::hash() const noexcept
{
  return hash_all(tid, fid, memory.id);
}

}  // namespace legate::mapping::detail
