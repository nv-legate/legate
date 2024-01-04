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

#include "core/mapping/mapping.h"
#include "core/utilities/detail/hash.h"
#include "core/utilities/hash.h"
#include "core/utilities/internal_shared_ptr.h"

#include <iosfwd>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

namespace legate::mapping::detail {

// This class represents a set of regions that colocate in an instance
struct RegionGroup {
 public:
  using Region = Legion::LogicalRegion;

  RegionGroup() = default;

  RegionGroup(std::set<Region> regions, const Domain& bounding_box);

  [[nodiscard]] std::vector<Region> get_regions() const;
  [[nodiscard]] bool subsumes(const RegionGroup* other);

  std::set<Region> regions{};
  Domain bounding_box{};
  std::unordered_map<const RegionGroup*, bool> subsumption_cache{};
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
  using Region       = Legion::LogicalRegion;
  using Instance     = Legion::Mapping::PhysicalInstance;
  using RegionGroupP = InternalSharedPtr<RegionGroup>;

  struct InstanceSpec {
    InstanceSpec() = default;

    InstanceSpec(Instance inst, InstanceMappingPolicy po);

    Instance instance{};
    InstanceMappingPolicy policy{};
  };

  [[nodiscard]] bool find_instance(Region region,
                                   Instance& result,
                                   const InstanceMappingPolicy& policy) const;
  [[nodiscard]] RegionGroupP construct_overlapping_region_group(const Region& region,
                                                                const Domain& domain,
                                                                bool exact) const;

  [[nodiscard]] std::set<Instance> record_instance(const RegionGroupP& group,
                                                   const Instance& instance,
                                                   const InstanceMappingPolicy& policy);

  bool erase(const Instance& inst);

  [[nodiscard]] size_t get_instance_size() const;

 private:
  void dump_and_sanity_check() const;

  std::unordered_map<RegionGroup*, InstanceSpec> instances_{};
  std::unordered_map<Region, RegionGroupP> groups_{};
};

class ReductionInstanceSet {
 public:
  using Region        = Legion::LogicalRegion;
  using Instance      = Legion::Mapping::PhysicalInstance;
  using ReductionOpID = Legion::ReductionOpID;

  struct ReductionInstanceSpec {
    ReductionInstanceSpec() = default;
    ReductionInstanceSpec(const ReductionOpID& op, Instance inst, InstanceMappingPolicy po);

    ReductionOpID redop{0};
    Instance instance{};
    InstanceMappingPolicy policy{};
  };

  [[nodiscard]] bool find_instance(ReductionOpID& redop,
                                   Region& region,
                                   Instance& result,
                                   const InstanceMappingPolicy& policy) const;

  void record_instance(ReductionOpID& redop,
                       Region& region,
                       Instance& instance,
                       const InstanceMappingPolicy& policy);

  bool erase(const Instance& inst);

 private:
  std::unordered_map<Region, ReductionInstanceSpec> instances_{};
};

class BaseInstanceManager {
 public:
  using Region       = Legion::LogicalRegion;
  using RegionTreeID = Legion::RegionTreeID;
  using Instance     = Legion::Mapping::PhysicalInstance;
  using FieldID      = Legion::FieldID;

  struct FieldMemInfo {
   public:
    FieldMemInfo(RegionTreeID t, FieldID f, Memory m);

    bool operator==(const FieldMemInfo& rhs) const;
    [[nodiscard]] size_t hash() const noexcept;

    RegionTreeID tid;
    FieldID fid;
    Memory memory;
  };

  [[nodiscard]] Legion::Mapping::LocalLock& manager_lock();

 private:
  Legion::Mapping::LocalLock manager_lock_{};
};

class InstanceManager : public BaseInstanceManager {
 public:
  using RegionGroupP = InternalSharedPtr<RegionGroup>;

  [[nodiscard]] bool find_instance(Region region,
                                   FieldID field_id,
                                   Memory memory,
                                   Instance& result,
                                   const InstanceMappingPolicy& policy = {});
  [[nodiscard]] RegionGroupP find_region_group(const Region& region,
                                               const Domain& domain,
                                               FieldID field_id,
                                               Memory memory,
                                               bool exact = false);
  void record_instance(const RegionGroupP& group,
                       FieldID field_id,
                       const Instance& instance,
                       const InstanceMappingPolicy& policy = {});

  void erase(const Instance& inst);

  void destroy();

  [[nodiscard]] static InstanceManager* get_instance_manager();

  [[nodiscard]] std::map<Memory, size_t> aggregate_instance_sizes() const;

 private:
  std::unordered_map<FieldMemInfo, InstanceSet, hasher<FieldMemInfo>> instance_sets_{};
};

class ReductionInstanceManager : public BaseInstanceManager {
 public:
  using ReductionOpID = Legion::ReductionOpID;

  [[nodiscard]] bool find_instance(ReductionOpID& redop,
                                   Region region,
                                   FieldID field_id,
                                   Memory memory,
                                   Instance& result,
                                   const InstanceMappingPolicy& policy = {});

  void record_instance(ReductionOpID& redop,
                       Region region,
                       FieldID field_id,
                       Instance instance,
                       const InstanceMappingPolicy& policy = {});

  void erase(const Instance& inst);

  void destroy();

  [[nodiscard]] static ReductionInstanceManager* get_instance_manager();

 private:
  std::unordered_map<FieldMemInfo, ReductionInstanceSet, hasher<FieldMemInfo>> instance_sets_{};
};

}  // namespace legate::mapping::detail

#include "core/mapping/detail/instance_manager.inl"
