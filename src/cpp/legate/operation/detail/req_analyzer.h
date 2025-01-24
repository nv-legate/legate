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

#include <legate/data/detail/logical_region_field.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/utilities/detail/hash.h>
#include <legate/utilities/hash.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace legate::detail {

class InterferingStoreError : public std::exception {};

class ProjectionSet {
 public:
  void insert(Legion::PrivilegeMode new_privilege,
              const StoreProjection& store_proj,
              bool relax_interference_checks);

  Legion::PrivilegeMode privilege{};
  std::set<BaseStoreProjection> store_projs{};
  bool is_key{};
};

class FieldSet {
 public:
  using Key = std::pair<Legion::PrivilegeMode, BaseStoreProjection>;

  void insert(Legion::FieldID field_id,
              Legion::PrivilegeMode privilege,
              const StoreProjection& store_proj,
              bool relax_interference_checks);
  [[nodiscard]] std::uint32_t num_requirements() const;
  [[nodiscard]] std::uint32_t get_requirement_index(Legion::PrivilegeMode privilege,
                                                    const StoreProjection& store_proj,
                                                    Legion::FieldID field_id) const;

  void coalesce();
  template <typename Launcher>
  void populate_launcher(Launcher* task, const Legion::LogicalRegion& region) const;

 private:
  class Entry {
   public:
    std::vector<Legion::FieldID> fields{};
    bool is_key{};
  };
  // This must be an ordered map to avoid control divergence
  std::map<Key, Entry> coalesced_{};
  using ReqIndexMapKey = std::pair<Key, Legion::FieldID>;
  std::unordered_map<ReqIndexMapKey, std::uint32_t, hasher<ReqIndexMapKey>> req_indices_{};

  // This must be an ordered map to avoid control divergence
  std::map<Legion::FieldID, ProjectionSet> field_projs_{};
};

class RequirementAnalyzer {
 public:
  void insert(const Legion::LogicalRegion& region,
              Legion::FieldID field_id,
              Legion::PrivilegeMode privilege,
              const StoreProjection& store_proj);
  [[nodiscard]] std::uint32_t get_requirement_index(const Legion::LogicalRegion& region,
                                                    Legion::PrivilegeMode privilege,
                                                    const StoreProjection& store_proj,
                                                    Legion::FieldID field_id) const;
  [[nodiscard]] bool empty() const;

  void analyze_requirements();
  void relax_interference_checks(bool relax);

  void populate_launcher(Legion::IndexTaskLauncher& task) const;
  void populate_launcher(Legion::TaskLauncher& task) const;

 private:
  template <typename Launcher>
  void populate_launcher_(Launcher* task) const;

  bool relax_interference_checks_{};
  // This must be an ordered map to avoid control divergence
  std::map<Legion::LogicalRegion, std::pair<FieldSet, std::uint32_t>> field_sets_{};
};

class OutputRequirementAnalyzer {
 public:
  void insert(std::uint32_t dim, const Legion::FieldSpace& field_space, Legion::FieldID field_id);
  [[nodiscard]] std::uint32_t get_requirement_index(const Legion::FieldSpace& field_space,
                                                    Legion::FieldID field_id) const;
  [[nodiscard]] bool empty() const;

  void analyze_requirements();
  void populate_output_requirements(std::vector<Legion::OutputRequirement>& out_reqs) const;

 private:
  class ReqInfo {
   public:
    static constexpr std::uint32_t UNSET = -1U;
    std::uint32_t dim{UNSET};
    std::uint32_t req_idx{};
  };
  // This must be an ordered map to avoid control divergence
  std::map<Legion::FieldSpace, std::set<Legion::FieldID>> field_groups_{};
  std::unordered_map<Legion::FieldSpace, ReqInfo> req_infos_{};
};

class FutureAnalyzer {
 public:
  void insert(Legion::Future future);
  void insert(Legion::FutureMap future_map);
  [[nodiscard]] std::int32_t get_index(const Legion::Future& future) const;
  [[nodiscard]] std::int32_t get_index(const Legion::FutureMap& future_map) const;

  void analyze_futures();
  void populate_launcher(Legion::IndexTaskLauncher& task) const;
  void populate_launcher(Legion::TaskLauncher& task) const;

 private:
  // XXX: This could be a hash map, but Legion futures don't reveal IDs that we can hash
  std::unordered_map<Legion::Future, std::int32_t> future_indices_{};
  std::unordered_map<Legion::FutureMap, std::int32_t> future_map_indices_{};
  std::vector<Legion::Future> coalesced_futures_{};
  std::vector<Legion::FutureMap> coalesced_future_maps_{};
  std::vector<Legion::Future> futures_{};
  std::vector<Legion::FutureMap> future_maps_{};
};

class StoreAnalyzer {
 public:
  void insert(const InternalSharedPtr<LogicalRegionField>& region_field,
              Legion::PrivilegeMode privilege,
              const StoreProjection& store_proj);
  void insert(std::uint32_t dim, const Legion::FieldSpace& field_space, Legion::FieldID field_id);
  void insert(Legion::Future future);
  void insert(Legion::FutureMap future_map);

  void analyze();

  [[nodiscard]] std::uint32_t get_index(const Legion::LogicalRegion& region,
                                        Legion::PrivilegeMode privilege,
                                        const StoreProjection& store_proj,
                                        Legion::FieldID field_id) const;
  [[nodiscard]] std::uint32_t get_index(const Legion::FieldSpace& field_space,
                                        Legion::FieldID field_id) const;
  [[nodiscard]] std::int32_t get_index(const Legion::Future& future) const;
  [[nodiscard]] std::int32_t get_index(const Legion::FutureMap& future_map) const;

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

#include <legate/operation/detail/req_analyzer.inl>
