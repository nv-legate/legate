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

#include "core/operation/detail/req_analyzer.h"

namespace legate::detail {

inline bool RequirementAnalyzer::empty() const { return field_sets_.empty(); }

// ==========================================================================================

inline bool OutputRequirementAnalyzer::empty() const { return field_groups_.empty(); }

// ==========================================================================================

inline void StoreAnalyzer::insert(const std::shared_ptr<LogicalRegionField>& region_field,
                                  Legion::PrivilegeMode privilege,
                                  const ProjectionInfo& proj_info)
{
  req_analyzer_.insert(region_field->region(), region_field->field_id(), privilege, proj_info);
}

inline void StoreAnalyzer::insert(int32_t dim,
                                  const Legion::FieldSpace& field_space,
                                  Legion::FieldID field_id)
{
  out_analyzer_.insert(dim, field_space, field_id);
}

inline void StoreAnalyzer::insert(const Legion::Future& future) { fut_analyzer_.insert(future); }

inline void StoreAnalyzer::analyze()
{
  req_analyzer_.analyze_requirements();
  out_analyzer_.analyze_requirements();
  fut_analyzer_.analyze_futures();
}

inline uint32_t StoreAnalyzer::get_index(const Legion::LogicalRegion& region,
                                         Legion::PrivilegeMode privilege,
                                         const ProjectionInfo& proj_info,
                                         Legion::FieldID field_id) const
{
  return req_analyzer_.get_requirement_index(region, privilege, proj_info, field_id);
}

inline uint32_t StoreAnalyzer::get_index(const Legion::FieldSpace& field_space,
                                         Legion::FieldID field_id) const
{
  return out_analyzer_.get_requirement_index(field_space, field_id);
}

inline int32_t StoreAnalyzer::get_index(const Legion::Future& future) const
{
  return fut_analyzer_.get_future_index(future);
}

template <typename Launcher>
inline void StoreAnalyzer::populate(Launcher& launcher,
                                    std::vector<Legion::OutputRequirement>& out_reqs)
{
  req_analyzer_.populate_launcher(launcher);
  out_analyzer_.populate_output_requirements(out_reqs);
  fut_analyzer_.populate_launcher(launcher);
}

inline bool StoreAnalyzer::can_be_local_function_task() const
{
  return req_analyzer_.empty() && out_analyzer_.empty();
}

}  // namespace legate::detail
