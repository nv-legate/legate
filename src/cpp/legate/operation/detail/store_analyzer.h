/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/data/detail/logical_region_field.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/utilities/detail/hash.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/hash.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace legate::detail {

class LEGATE_EXPORT InterferingStoreError : public std::exception {};

class ProjectionSet {
 public:
  void insert(Legion::PrivilegeMode new_privilege,
              const StoreProjection& store_proj,
              bool relax_interference_checks);

  /**
   * @return The coalesced privilege of the projection set.
   *
   * @note This is not the true coalesced privilege of the projection set. During privilege
   * promotion, any discard-output masks (inserted as a result of streaming) are stripped out
   * of the privilege. In order to reconstruct the full true privilege, the caller must perform:
   *
   * ```c++
   * auto priv = proj_set.privilege();
   *
   * if (proj_set.had_streaming_discard()) {
   *   priv |= LEGION_DISCARD_OUTPUT_MASK;
   * }
   * ```
   */
  [[nodiscard]] Legion::PrivilegeMode privilege() const;

  /**
   * @return The store projections.
   */
  [[nodiscard]] const std::set<BaseStoreProjection>& store_projs() const;

  /**
   * @return Whether this projection set corresponds to the key store.
   */
  [[nodiscard]] bool is_key() const;

  /**
   * @return Whether this projection set had the streaming discard privilege privilege stripped
   * out during privilege promotion.
   */
  [[nodiscard]] bool had_streaming_discard() const;

 private:
  bool had_streaming_discard_{};
  Legion::PrivilegeMode privilege_{};
  std::set<BaseStoreProjection> store_projs_{};
  bool is_key_{};
};

class FieldSet {
 public:
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
  // Note the privilege in this "Key" is not the true promoted privilege of the store. If the
  // incoming store had the streaming discard mask (discard-output), then this mask will have
  // been stripped out during privilege promotion. The true promoted privilege will be `priv |
  // LEGION_DISCARD_OUTPUT_MASK`, if `Entry::has_streaming_discard_` is true, otherwise just
  // `priv`.
  using Key = std::pair<Legion::PrivilegeMode, BaseStoreProjection>;

  class Entry {
   public:
    std::vector<Legion::FieldID> fields{};
    bool is_key{};
    bool has_streaming_discard_{};
  };
  // This must be an ordered map to avoid control divergence.
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
  SmallVector<Legion::Future> coalesced_futures_{};
  SmallVector<Legion::FutureMap> coalesced_future_maps_{};
  SmallVector<Legion::Future> futures_{};
  SmallVector<Legion::FutureMap> future_maps_{};
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
  void populate(Launcher& launcher, std::vector<Legion::OutputRequirement>& out_reqs) const;

  [[nodiscard]] bool can_be_local_function_task() const;
  void relax_interference_checks(bool relax);

 private:
  RequirementAnalyzer req_analyzer_{};
  OutputRequirementAnalyzer out_analyzer_{};
  FutureAnalyzer fut_analyzer_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/store_analyzer.inl>
