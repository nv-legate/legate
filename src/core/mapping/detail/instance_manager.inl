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

#include "core/mapping/detail/instance_manager.h"

namespace legate::mapping::detail {

inline RegionGroup::RegionGroup(std::set<Region> regions, Domain bounding_box)
  : regions{std::move(regions)}, bounding_box{bounding_box}
{
}

inline std::vector<RegionGroup::Region> RegionGroup::get_regions() const
{
  return {regions.begin(), regions.end()};
}

// ==========================================================================================

inline InstanceSet::InstanceSpec::InstanceSpec(Instance inst, InstanceMappingPolicy po)
  : instance{std::move(inst)}, policy{std::move(po)}
{
}

// ==========================================================================================

inline ReductionInstanceSet::ReductionInstanceSpec::ReductionInstanceSpec(const ReductionOpID& op,
                                                                          Instance inst,
                                                                          InstanceMappingPolicy po)
  : redop{op}, instance{std::move(inst)}, policy{std::move(po)}
{
}

// ==========================================================================================

inline BaseInstanceManager::FieldMemInfo::FieldMemInfo(RegionTreeID t, FieldID f, Memory m)
  : tid{t}, fid{f}, memory{m}
{
}

inline bool BaseInstanceManager::FieldMemInfo::operator==(const FieldMemInfo& rhs) const
{
  return tid == rhs.tid && fid == rhs.fid && memory == rhs.memory;
}

inline bool BaseInstanceManager::FieldMemInfo::operator<(const FieldMemInfo& rhs) const
{
  if (tid < rhs.tid) return true;
  if (tid > rhs.tid) return false;
  if (fid < rhs.fid) return true;
  if (fid > rhs.fid) return false;
  return memory < rhs.memory;
}

inline Legion::Mapping::LocalLock& BaseInstanceManager::manager_lock() { return manager_lock_; }

}  // namespace legate::mapping::detail
