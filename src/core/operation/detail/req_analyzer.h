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
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

class ProjectionSet {
 public:
  void insert(Legion::PrivilegeMode new_privilege, const ProjectionInfo& proj_info);

 public:
  Legion::PrivilegeMode privilege;
  std::set<ProjectionInfo> proj_infos;
};

class FieldSet {
 public:
  using Key = std::pair<Legion::PrivilegeMode, ProjectionInfo>;

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
  void populate_launcher(Launcher* task, const Legion::LogicalRegion& region) const;

 private:
  std::map<Key, std::vector<Legion::FieldID>> coalesced_;
  std::map<Key, uint32_t> req_indices_;

 private:
  std::map<Legion::FieldID, ProjectionSet> field_projs_;
};

class RequirementAnalyzer {
 public:
  ~RequirementAnalyzer();

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
  void populate_launcher(Legion::IndexTaskLauncher* task) const;
  void populate_launcher(Legion::TaskLauncher* task) const;

 private:
  template <class Launcher>
  void _populate_launcher(Launcher* task) const;

 private:
  std::map<Legion::LogicalRegion, std::pair<FieldSet, uint32_t>> field_sets_;
};

class OutputRequirementAnalyzer {
 public:
  ~OutputRequirementAnalyzer();

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
  std::map<Legion::FieldSpace, std::set<Legion::FieldID>> field_groups_;
  std::map<Legion::FieldSpace, ReqInfo> req_infos_;
};

}  // namespace legate::detail
