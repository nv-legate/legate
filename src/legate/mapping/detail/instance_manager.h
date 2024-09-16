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

#include "legate/mapping/mapping.h"
#include "legate/utilities/detail/hash.h"
#include "legate/utilities/hash.h"
#include "legate/utilities/internal_shared_ptr.h"

#include <iosfwd>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>

namespace legate::mapping::detail {

// This class represents a set of regions that colocate in an instance
class RegionGroup : public EnableSharedFromThis<RegionGroup> {
 public:
  RegionGroup() = default;

  RegionGroup(std::set<Legion::LogicalRegion> regions, const Domain& bounding_box);

  std::set<Legion::LogicalRegion> regions{};
  Domain bounding_box{};
};

std::ostream& operator<<(std::ostream& os, const RegionGroup& region_group);

// FIXME: If clang-tidy ever lets us ignore warnings from specific headers, or fixes this, we
// can remove the NOLINT. This NOLINT is added to work around a bogus clang-tidy warning:
//
// _deps/legion-src/runtime/legion/legion_types.h:2027:11: error: no definition found for
// 'InstanceSet', but a definition with the same name 'InstanceSet' found in another namespace
// 'legate::mapping::detail' [bugprone-forward-declaration-namespace,-warnings-as-errors]
// 2027 | class InstanceSet;
//      |       ^
// legate.core.internal/src/core/mapping/detail/instance_manager.h:44:8: note: a definition of
// 'InstanceSet' is found here
//   44 | class InstanceSet {
//      |       ^
//
// The only way (other than to disable the check wholesale), is to silence it for this class...
class InstanceSet {  // NOLINT(bugprone-forward-declaration-namespace)
 public:
  class InstanceSpec {
   public:
    InstanceSpec() = default;

    InstanceSpec(Legion::Mapping::PhysicalInstance inst, InstanceMappingPolicy po);

    Legion::Mapping::PhysicalInstance instance{};
    InstanceMappingPolicy policy{};
  };

  [[nodiscard]] std::optional<Legion::Mapping::PhysicalInstance> find_instance(
    const Legion::LogicalRegion& region, const InstanceMappingPolicy& policy) const;
  [[nodiscard]] InternalSharedPtr<RegionGroup> find_or_create_region_group(
    const Legion::LogicalRegion& region, const Domain& domain, bool exact) const;

  void record_instance(const Legion::LogicalRegion& region,
                       const InternalSharedPtr<RegionGroup>& group,
                       Legion::Mapping::PhysicalInstance instance,
                       InstanceMappingPolicy policy);
  void record_pending_instance_creation(InternalSharedPtr<RegionGroup> group);
  void remove_pending_instance(const InternalSharedPtr<RegionGroup>& group);

  [[nodiscard]] bool empty() const;

  [[nodiscard]] bool erase(const Legion::Mapping::PhysicalInstance& inst);

  [[nodiscard]] std::size_t get_instance_size() const;

 private:
  void dump_and_sanity_check_() const;

  std::unordered_map<RegionGroup*, InstanceSpec> instances_{};
  std::unordered_map<InternalSharedPtr<RegionGroup>, std::uint64_t> pending_instances_{};
  std::unordered_map<Legion::LogicalRegion, InternalSharedPtr<RegionGroup>> groups_{};
};

class ReductionInstanceSet {
 public:
  class ReductionInstanceSpec {
   public:
    ReductionInstanceSpec() = default;
    ReductionInstanceSpec(GlobalRedopID op,
                          Legion::Mapping::PhysicalInstance inst,
                          InstanceMappingPolicy po);

    GlobalRedopID redop{0};
    Legion::Mapping::PhysicalInstance instance{};
    InstanceMappingPolicy policy{};
  };

  [[nodiscard]] std::optional<Legion::Mapping::PhysicalInstance> find_instance(
    GlobalRedopID redop,
    const Legion::LogicalRegion& region,
    const InstanceMappingPolicy& policy) const;

  void record_instance(GlobalRedopID redop,
                       const Legion::LogicalRegion& region,
                       Legion::Mapping::PhysicalInstance instance,
                       InstanceMappingPolicy policy);

  [[nodiscard]] bool empty() const;

  [[nodiscard]] bool erase(const Legion::Mapping::PhysicalInstance& inst);

 private:
  std::unordered_map<Legion::LogicalRegion, ReductionInstanceSpec> instances_{};
};

class BaseInstanceManager {
 public:
  class FieldMemInfo {
   public:
    FieldMemInfo(Legion::RegionTreeID t, Legion::FieldID f, Memory m);

    [[nodiscard]] bool operator==(const FieldMemInfo& rhs) const;
    [[nodiscard]] std::size_t hash() const noexcept;

    Legion::RegionTreeID tid{};
    Legion::FieldID fid{};
    Memory memory{};
  };

 protected:
  template <typename T>
  [[nodiscard]] static bool do_erase_(
    std::unordered_map<FieldMemInfo, T, hasher<FieldMemInfo>>* instance_sets,
    const Legion::Mapping::PhysicalInstance& inst);
};

class InstanceManager final : public BaseInstanceManager {
 public:
  [[nodiscard]] std::optional<Legion::Mapping::PhysicalInstance> find_instance(
    const Legion::LogicalRegion& region,
    Legion::FieldID field_id,
    Memory memory,
    const InstanceMappingPolicy& policy = {});
  [[nodiscard]] InternalSharedPtr<RegionGroup> find_region_group(
    const Legion::LogicalRegion& region,
    const Domain& domain,
    Legion::FieldID field_id,
    Memory memory,
    bool exact = false);
  void record_instance(const Legion::LogicalRegion& region,
                       const InternalSharedPtr<RegionGroup>& group,
                       Legion::FieldID field_id,
                       Legion::Mapping::PhysicalInstance instance,
                       InstanceMappingPolicy policy = {});
  void remove_pending_instance(const Legion::LogicalRegion& region,
                               const InternalSharedPtr<RegionGroup>& group,
                               Legion::FieldID field_id,
                               const Memory& memory);

  [[nodiscard]] bool erase(const Legion::Mapping::PhysicalInstance& inst);

  [[nodiscard]] std::map<Memory, std::size_t> aggregate_instance_sizes() const;

 private:
  std::unordered_map<FieldMemInfo, InstanceSet, hasher<FieldMemInfo>> instance_sets_{};
};

class ReductionInstanceManager final : public BaseInstanceManager {
 public:
  [[nodiscard]] std::optional<Legion::Mapping::PhysicalInstance> find_instance(
    GlobalRedopID redop,
    const Legion::LogicalRegion& region,
    Legion::FieldID field_id,
    Memory memory,
    const InstanceMappingPolicy& policy = {});

  void record_instance(GlobalRedopID redop,
                       const Legion::LogicalRegion& region,
                       Legion::FieldID field_id,
                       Legion::Mapping::PhysicalInstance instance,
                       InstanceMappingPolicy policy = {});

  [[nodiscard]] bool erase(const Legion::Mapping::PhysicalInstance& inst);

 private:
  std::unordered_map<FieldMemInfo, ReductionInstanceSet, hasher<FieldMemInfo>> instance_sets_{};
};

}  // namespace legate::mapping::detail

#include "legate/mapping/detail/instance_manager.inl"
