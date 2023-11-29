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

#include <map>
#include <set>
#include <utility>
#include <vector>

namespace legate::detail {

struct InterferingStoreError : public std::exception {};

class ProjectionSet {
 public:
  void insert(Legion::PrivilegeMode new_privilege,
              const ProjectionInfo& proj_info,
              bool relax_interference_checks);

  Legion::PrivilegeMode privilege{};
  std::set<BaseProjectionInfo> proj_infos{};
  bool is_key{};
};

class FieldSet {
 public:
  using Key = std::pair<Legion::PrivilegeMode, BaseProjectionInfo>;

  void insert(Legion::FieldID field_id,
              Legion::PrivilegeMode privilege,
              const ProjectionInfo& proj_info,
              bool relax_interference_checks);
  [[nodiscard]] uint32_t num_requirements() const;
  [[nodiscard]] uint32_t get_requirement_index(Legion::PrivilegeMode privilege,
                                               const ProjectionInfo& proj_info,
                                               Legion::FieldID field_id) const;

  void coalesce();
  template <class Launcher>
  void populate_launcher(Launcher& task, const Legion::LogicalRegion& region) const;

 private:
  struct Entry {
    std::vector<Legion::FieldID> fields{};
    bool is_key{};
  };
  std::map<Key, Entry> coalesced_{};
  std::map<std::pair<Key, Legion::FieldID>, uint32_t> req_indices_{};

  std::map<Legion::FieldID, ProjectionSet> field_projs_{};
};

class RequirementAnalyzer {
 public:
  void insert(const Legion::LogicalRegion& region,
              Legion::FieldID field_id,
              Legion::PrivilegeMode privilege,
              const ProjectionInfo& proj_info);
  [[nodiscard]] uint32_t get_requirement_index(const Legion::LogicalRegion& region,
                                               Legion::PrivilegeMode privilege,
                                               const ProjectionInfo& proj_info,
                                               Legion::FieldID field_id) const;
  [[nodiscard]] bool empty() const;

  void analyze_requirements();
  void relax_interference_checks(bool relax);

  void populate_launcher(Legion::IndexTaskLauncher& task) const;
  void populate_launcher(Legion::TaskLauncher& task) const;

 private:
  template <class Launcher>
  void _populate_launcher(Launcher& task) const;

  bool relax_interference_checks_{};
  std::map<Legion::LogicalRegion, std::pair<FieldSet, uint32_t>> field_sets_{};
};

class OutputRequirementAnalyzer {
 public:
  void insert(int32_t dim, const Legion::FieldSpace& field_space, Legion::FieldID field_id);
  [[nodiscard]] uint32_t get_requirement_index(const Legion::FieldSpace& field_space,
                                               Legion::FieldID field_id) const;
  [[nodiscard]] bool empty() const;

  void analyze_requirements();
  void populate_output_requirements(std::vector<Legion::OutputRequirement>& out_reqs) const;

 private:
  struct ReqInfo {
    int32_t dim{-1};
    uint32_t req_idx{};
  };
  std::map<Legion::FieldSpace, std::set<Legion::FieldID>> field_groups_{};
  std::map<Legion::FieldSpace, ReqInfo> req_infos_{};
};

class FutureAnalyzer {
 public:
  void insert(const Legion::Future& future);
  [[nodiscard]] int32_t get_future_index(const Legion::Future& future) const;

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
  void insert(const std::shared_ptr<LogicalRegionField>& region_field,
              Legion::PrivilegeMode privilege,
              const ProjectionInfo& proj_info);
  void insert(int32_t dim, const Legion::FieldSpace& field_space, Legion::FieldID field_id);
  void insert(const Legion::Future& future);

  void analyze();

  [[nodiscard]] uint32_t get_index(const Legion::LogicalRegion& region,
                                   Legion::PrivilegeMode privilege,
                                   const ProjectionInfo& proj_info,
                                   Legion::FieldID field_id) const;
  [[nodiscard]] uint32_t get_index(const Legion::FieldSpace& field_space,
                                   Legion::FieldID field_id) const;
  [[nodiscard]] int32_t get_index(const Legion::Future& future) const;

  template <typename Launcher>
  void populate(Launcher& launcher, std::vector<Legion::OutputRequirement>& out_reqs);

  [[nodiscard]] bool can_be_local_function_task() const;
  void relax_interference_checks(bool relax);

 private:
  RequirementAnalyzer req_analyzer_{};
  OutputRequirementAnalyzer out_analyzer_{};
  FutureAnalyzer fut_analyzer_{};
};

}  // namespace legate::detail

#include "core/operation/detail/req_analyzer.inl"
