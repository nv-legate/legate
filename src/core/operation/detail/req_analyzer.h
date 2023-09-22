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

#include "core/data/detail/logical_region_field.h"
#include "core/operation/detail/projection.h"
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

class ProjectionSet {
 public:
  void insert(Legion::PrivilegeMode new_privilege, const ProjectionInfo& proj_info);

 public:
  Legion::PrivilegeMode privilege;
  std::set<BaseProjectionInfo> proj_infos;
  bool is_key;
};

class FieldSet {
 public:
  using Key = std::pair<Legion::PrivilegeMode, BaseProjectionInfo>;

 public:
  void insert(Legion::FieldID field_id,
              Legion::PrivilegeMode privilege,
              const ProjectionInfo& proj_info);
  uint32_t num_requirements() const;
  uint32_t get_requirement_index(Legion::PrivilegeMode privilege,
                                 const ProjectionInfo& proj_info) const;

 public:
  void coalesce();
  template <class Launcher>
  void populate_launcher(Launcher& task, const Legion::LogicalRegion& region) const;

 private:
  struct Entry {
    std::vector<Legion::FieldID> fields;
    bool is_key;
  };
  std::map<Key, Entry> coalesced_;
  std::map<Key, uint32_t> req_indices_;

 private:
  std::map<Legion::FieldID, ProjectionSet> field_projs_;
};

class RequirementAnalyzer {
 public:
  void insert(const Legion::LogicalRegion& region,
              Legion::FieldID field_id,
              Legion::PrivilegeMode privilege,
              const ProjectionInfo& proj_info);
  uint32_t get_requirement_index(const Legion::LogicalRegion& region,
                                 Legion::PrivilegeMode privilege,
                                 const ProjectionInfo& proj_info) const;
  bool empty() const { return field_sets_.empty(); }

 public:
  void analyze_requirements();
  void populate_launcher(Legion::IndexTaskLauncher& task) const;
  void populate_launcher(Legion::TaskLauncher& task) const;

 private:
  template <class Launcher>
  void _populate_launcher(Launcher& task) const;

 private:
  std::map<Legion::LogicalRegion, std::pair<FieldSet, uint32_t>> field_sets_{};
};

class OutputRequirementAnalyzer {
 public:
  void insert(int32_t dim, const Legion::FieldSpace& field_space, Legion::FieldID field_id);
  uint32_t get_requirement_index(const Legion::FieldSpace& field_space,
                                 Legion::FieldID field_id) const;
  bool empty() const { return field_groups_.empty(); }

 public:
  void analyze_requirements();
  void populate_output_requirements(std::vector<Legion::OutputRequirement>& out_reqs) const;

 private:
  struct ReqInfo {
    int32_t dim{-1};
    uint32_t req_idx{0};
  };
  std::map<Legion::FieldSpace, std::set<Legion::FieldID>> field_groups_{};
  std::map<Legion::FieldSpace, ReqInfo> req_infos_{};
};

class FutureAnalyzer {
 public:
  void insert(const Legion::Future& future);
  int32_t get_future_index(const Legion::Future& future) const;

 public:
  void analyze_futures();
  template <class Launcher>
  void _populate_launcher(Launcher& task) const;
  void populate_launcher(Legion::IndexTaskLauncher& task) const;
  void populate_launcher(Legion::TaskLauncher& task) const;

 private:
  std::map<Legion::Future, int32_t> future_indices_{};
  std::vector<Legion::Future> coalesced_{};
  std::vector<Legion::Future> futures_{};
};

struct StoreAnalyzer {
 public:
  void insert(const LogicalRegionField* region_field,
              Legion::PrivilegeMode privilege,
              const ProjectionInfo& proj_info)
  {
    req_analyzer_.insert(region_field->region(), region_field->field_id(), privilege, proj_info);
  }
  void insert(int32_t dim, const Legion::FieldSpace& field_space, Legion::FieldID field_id)
  {
    out_analyzer_.insert(dim, field_space, field_id);
  }
  void insert(const Legion::Future& future) { fut_analyzer_.insert(future); }

 public:
  void analyze()
  {
    req_analyzer_.analyze_requirements();
    out_analyzer_.analyze_requirements();
    fut_analyzer_.analyze_futures();
  }

 public:
  uint32_t get_index(const Legion::LogicalRegion& region,
                     Legion::PrivilegeMode privilege,
                     const ProjectionInfo& proj_info) const
  {
    return req_analyzer_.get_requirement_index(region, privilege, proj_info);
  }
  uint32_t get_index(const Legion::FieldSpace& field_space, Legion::FieldID field_id) const
  {
    return out_analyzer_.get_requirement_index(field_space, field_id);
  }
  int32_t get_index(const Legion::Future& future) const
  {
    return fut_analyzer_.get_future_index(future);
  }

 public:
  template <typename Launcher>
  void populate(Launcher& launcher, std::vector<Legion::OutputRequirement>& out_reqs)
  {
    req_analyzer_.populate_launcher(launcher);
    out_analyzer_.populate_output_requirements(out_reqs);
    fut_analyzer_.populate_launcher(launcher);
  }

 public:
  bool can_be_local_function_task() const { return req_analyzer_.empty() && out_analyzer_.empty(); }

 private:
  RequirementAnalyzer req_analyzer_{};
  OutputRequirementAnalyzer out_analyzer_{};
  FutureAnalyzer fut_analyzer_{};
};

}  // namespace legate::detail
