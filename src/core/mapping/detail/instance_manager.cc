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

#include "core/mapping/detail/instance_manager.h"

#include "core/utilities/dispatch.h"
#include "core/utilities/internal_shared_ptr.h"

#include <algorithm>
#include <iostream>
#include <unordered_set>

namespace legate::mapping::detail {

namespace {

[[nodiscard]] Legion::Logger& log_instmgr()
{
  static Legion::Logger log{"instmgr"};

  return log;
}

}  // namespace

bool RegionGroup::subsumes(const RegionGroup* other)
{
  if (regions.size() < other->regions.size()) {
    return false;
  }
  if (other->regions.size() == 1) {
    return regions.find(*other->regions.begin()) != regions.end();
  }

  if (const auto it = subsumption_cache.find(other); it != subsumption_cache.end()) {
    return it->second;
  }
  for (auto&& region : other->regions) {
    if (regions.find(region) == regions.end()) {
      subsumption_cache[other] = false;
      return false;
    }
  }

  subsumption_cache[other] = true;
  return true;
}

std::ostream& operator<<(std::ostream& os, const RegionGroup& region_group)
{
  os << "RegionGroup(" << region_group.bounding_box << ": {";
  for (const auto& region : region_group.regions) {
    os << region << ",";
  }
  os << "})";
  return os;
}

std::optional<Legion::Mapping::PhysicalInstance> InstanceSet::find_instance(
  const Legion::LogicalRegion& region, const InstanceMappingPolicy& policy) const
{
  const auto finder = groups_.find(region);

  if (finder == groups_.end()) {
    return std::nullopt;
  }

  auto&& group = finder->second;

  if (policy.exact && group->regions.size() > 1) {
    return std::nullopt;
  }

  const auto ifinder = instances_.find(group.get());
  LEGATE_CHECK(ifinder != instances_.end());

  if (auto&& spec = ifinder->second; spec.policy.subsumes(policy)) {
    return spec.instance;
  }
  return std::nullopt;
}

namespace {

// We define "too big" as the size of the "unused" points being bigger than the intersection
bool too_big(std::size_t union_volume,
             std::size_t my_volume,
             std::size_t group_volume,
             std::size_t intersect_volume)
{
  return (union_volume - (my_volume + group_volume - intersect_volume)) > intersect_volume;
}

class ConstructOverlappingRegionGroupFn {
 public:
  template <std::int32_t DIM>
  InternalSharedPtr<RegionGroup> operator()(
    const Legion::LogicalRegion& region,
    const Domain& domain,
    const std::unordered_map<RegionGroup*, InstanceSet::InstanceSpec>& instances)
  {
    auto bound            = domain.bounds<DIM, coord_t>();
    std::size_t bound_vol = bound.volume();
    std::set<Legion::LogicalRegion> regions{region};

    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      log_instmgr().debug() << "construct_overlapping_region_group( " << region << "," << domain
                            << ")";
    }

    for (const auto& pair : instances) {
      auto& group = pair.first;

      const Rect<DIM> group_bbox = group->bounding_box.bounds<DIM, coord_t>();
      if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
        log_instmgr().debug() << "  check intersection with " << group_bbox;
      }
      auto intersect = bound.intersection(group_bbox);
      if (intersect.empty()) {
        if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
          log_instmgr().debug() << "    no intersection";
        }
        continue;
      }

      // Only allow merging if the bloating isn't "too big"
      auto union_bbox                 = bound.union_bbox(group_bbox);
      const std::size_t union_vol     = union_bbox.volume();
      const std::size_t group_vol     = group_bbox.volume();
      const std::size_t intersect_vol = intersect.volume();
      if (too_big(union_vol, bound_vol, group_vol, intersect_vol)) {
        if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
          log_instmgr().debug() << "    too big to merge (union:" << union_bbox
                                << ",bound:" << bound_vol << ",group:" << group_vol
                                << ",intersect:" << intersect_vol << ")";
        }
        continue;
      }

      // NOTE: It is critical that we maintain the invariant that if at least one region is mapped
      // to a group in the instances_ table, that group is still present on the groups_ table, and
      // thus there's at least one shared_ptr remaining that points to it. Otherwise we run the risk
      // that a group pointer stored on the instances_ table points to a group that's been collected
      regions.insert(group->regions.begin(), group->regions.end());
      if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
        log_instmgr().debug() << "    bounds updated: " << bound << " ~> " << union_bbox;
      }

      bound     = union_bbox;
      bound_vol = union_vol;
    }

    return make_internal_shared<RegionGroup>(std::move(regions), Domain(bound));
  }
};

}  // namespace

InternalSharedPtr<RegionGroup> InstanceSet::construct_overlapping_region_group(
  const Legion::LogicalRegion& region, const Domain& domain, bool exact) const
{
  auto finder = groups_.find(region);
  if (finder == groups_.end()) {
    return dim_dispatch(
      domain.get_dim(), ConstructOverlappingRegionGroupFn{}, region, domain, instances_);
  }

  if (!exact || finder->second->regions.size() == 1) {
    return finder->second;
  }
  return make_internal_shared<RegionGroup>(std::set<Legion::LogicalRegion>{region}, domain);
}

void InstanceSet::record_instance(const InternalSharedPtr<RegionGroup>& group,
                                  Legion::Mapping::PhysicalInstance instance,
                                  InstanceMappingPolicy policy)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_instmgr().debug() << "===== before adding an entry " << *group << " ~> " << instance
                          << " =====";
  }
  dump_and_sanity_check_();

  const auto& inst =
    instances_.insert_or_assign(group.get(), InstanceSpec{std::move(instance), std::move(policy)})
      .first->second.instance;

  // Use of InternalSharedPtr vs raw RegionGroup * is deliberate. We swap the group down below,
  // and if the old region group is the last one left, we should delete it until we can remove
  // it from our instances_...
  std::unordered_set<InternalSharedPtr<RegionGroup>> removed_groups;

  for (auto&& region : group->regions) {
    const auto [it, inserted] = groups_.try_emplace(region, group);
    auto& old_group           = it->second;

    if (!inserted && (old_group != group)) {
      // ... swapped out here...
      removed_groups.emplace(old_group);
      old_group = group;
    }
  }

  for (auto&& removed_group : removed_groups) {
    // Because of exact policies, we can't simply remove the groups where regions in the `group`
    // originally belonged, because one region can be included in multiple region groups. (Note that
    // the exact mapping bypasses the coalescing heuristic and always creates a fresh singleton
    // group.) So, before we prune out each of those potentially obsolete groups, we need to
    // make sure that it has no remaining references.
    const auto can_remove = std::none_of(
      removed_group->regions.cbegin(),
      removed_group->regions.cend(),
      [&](const Legion::LogicalRegion& rg) { return groups_.at(rg) == removed_group; });

    if (can_remove) {
      // ... and erased here
      instances_.erase(removed_group.get());
    }
  }

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_instmgr().debug() << "===== after adding an entry " << *group << " ~> " << inst << " =====";
    dump_and_sanity_check_();
  }
}

bool InstanceSet::erase(const Legion::Mapping::PhysicalInstance& inst)
{
  std::set<RegionGroup*> filtered_groups;
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_instmgr().debug() << "===== before erasing an instance " << inst << " =====";
  }
  dump_and_sanity_check_();

  for (auto it = instances_.begin(); it != instances_.end(); /*nothing*/) {
    if (it->second.instance == inst) {
      filtered_groups.insert(it->first);
      it = instances_.erase(it);
    } else {
      ++it;
    }
  }

  std::set<Legion::LogicalRegion> filtered_regions;
  for (const RegionGroup* group : filtered_groups) {
    for (auto&& region : group->regions) {
      if (groups_.at(region).get() == group) {
        // We have to do this in two steps; we don't want to remove the last shared_ptr to a group
        // while iterating over the same group's regions
        filtered_regions.insert(region);
      }
    }
  }
  for (auto&& region : filtered_regions) {
    groups_.erase(region);
  }

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_instmgr().debug() << "===== after erasing an instance " << inst << " =====";
  }
  dump_and_sanity_check_();

  return instances_.empty();
}

std::size_t InstanceSet::get_instance_size() const
{
  std::size_t sum = 0;
  for (auto&& [_, instance_spec] : instances_) {
    sum += instance_spec.instance.get_instance_size();
  }
  return sum;
}

void InstanceSet::dump_and_sanity_check_() const
{
#ifdef DEBUG_INSTANCE_MANAGER
  for (auto&& entry : groups_) {
    log_instmgr().debug() << "  " << entry.first << " ~> " << *entry.second;
  }
  for (auto&& entry : instances_) {
    log_instmgr().debug() << "  " << *entry.first << " ~> " << entry.second.instance;
  }
#endif
  std::unordered_set<RegionGroup*> found_groups;
  for (auto&& entry : groups_) {
    found_groups.insert(entry.second.get());
    LEGATE_CHECK(instances_.count(entry.second.get()) > 0);
    LEGATE_CHECK(entry.second->regions.count(entry.first) > 0);
  }
  for (auto&& entry : instances_) {
    LEGATE_CHECK(found_groups.count(entry.first) > 0);
  }
}

std::optional<Legion::Mapping::PhysicalInstance> ReductionInstanceSet::find_instance(
  GlobalRedopID redop,
  const Legion::LogicalRegion& region,
  const InstanceMappingPolicy& policy) const
{
  if (const auto it = instances_.find(region); it != instances_.end()) {
    if (auto&& spec = it->second; spec.policy == policy && spec.redop == redop) {
      return spec.instance;
    }
  }
  return std::nullopt;
}

void ReductionInstanceSet::record_instance(GlobalRedopID redop,
                                           const Legion::LogicalRegion& region,
                                           Legion::Mapping::PhysicalInstance instance,
                                           InstanceMappingPolicy policy)
{
  if (const auto [it, inserted] =
        instances_.try_emplace(region, redop, std::move(instance), std::move(policy));
      !inserted) {
    if (auto&& spec = it->second; spec.policy != policy || spec.redop != redop) {
      spec = ReductionInstanceSpec{redop, std::move(instance), std::move(policy)};
    }
  }
}

bool ReductionInstanceSet::erase(const Legion::Mapping::PhysicalInstance& inst)
{
  for (auto it = instances_.begin(); it != instances_.end(); /*nothing*/) {
    if (it->second.instance == inst) {
      it = instances_.erase(it);
    } else {
      ++it;
    }
  }
  return instances_.empty();
}

std::optional<Legion::Mapping::PhysicalInstance> InstanceManager::find_instance(
  const Legion::LogicalRegion& region,
  Legion::FieldID field_id,
  Memory memory,
  const InstanceMappingPolicy& policy)
{
  if (policy.allocation == AllocPolicy::MUST_ALLOC) {
    return std::nullopt;
  }

  const auto it = instance_sets_.find({region.get_tree_id(), field_id, memory});

  if (it == instance_sets_.end()) {
    return std::nullopt;
  }

  return it->second.find_instance(region, policy);
}

InternalSharedPtr<RegionGroup> InstanceManager::find_region_group(
  const Legion::LogicalRegion& region,
  const Domain& domain,
  Legion::FieldID field_id,
  Memory memory,
  bool exact /*=false*/)
{
  InternalSharedPtr<RegionGroup> result = [&] {
    const auto it = instance_sets_.find({region.get_tree_id(), field_id, memory});

    if (it == instance_sets_.end() || exact) {
      return make_internal_shared<RegionGroup>(std::set<Legion::LogicalRegion>{region}, domain);
    }
    return it->second.construct_overlapping_region_group(region, domain, exact);
  }();

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_instmgr().debug() << "find_region_group(" << region << "," << domain << "," << field_id
                          << "," << memory << "," << exact << ") ~> " << *result;
  }

  return result;
}

void InstanceManager::record_instance(const InternalSharedPtr<RegionGroup>& group,
                                      Legion::FieldID field_id,
                                      Legion::Mapping::PhysicalInstance instance,
                                      InstanceMappingPolicy policy)
{
  FieldMemInfo key{instance.get_tree_id(), field_id, instance.get_location()};

  instance_sets_[std::move(key)].record_instance(group, std::move(instance), std::move(policy));
}

void InstanceManager::erase(const Legion::Mapping::PhysicalInstance& inst)
{
  const auto mem = inst.get_location();
  const auto tid = inst.get_tree_id();

  for (auto fit = instance_sets_.begin(); fit != instance_sets_.end(); /*nothing*/) {
    if ((fit->first.memory != mem) || (fit->first.tid != tid)) {
      ++fit;
      continue;
    }

    if (fit->second.erase(inst)) {
      fit = instance_sets_.erase(fit);
    } else {
      ++fit;
    }
  }
}

std::map<Memory, std::size_t> InstanceManager::aggregate_instance_sizes() const
{
  std::map<Memory, std::size_t> result;

  for (auto&& [field_mem_info, instance_set] : instance_sets_) {
    result[field_mem_info.memory] += instance_set.get_instance_size();
  }
  return result;
}

/*static*/ InstanceManager* InstanceManager::get_instance_manager()
{
  static InstanceManager manager{};
  return &manager;
}

std::optional<Legion::Mapping::PhysicalInstance> ReductionInstanceManager::find_instance(
  GlobalRedopID redop,
  const Legion::LogicalRegion& region,
  Legion::FieldID field_id,
  Memory memory,
  const InstanceMappingPolicy& policy)
{
  if (policy.allocation == AllocPolicy::MUST_ALLOC) {
    return std::nullopt;
  }

  const auto it = instance_sets_.find({region.get_tree_id(), field_id, memory});

  if (it == instance_sets_.end()) {
    return std::nullopt;
  }

  return it->second.find_instance(redop, region, policy);
}

void ReductionInstanceManager::record_instance(GlobalRedopID redop,
                                               const Legion::LogicalRegion& region,
                                               Legion::FieldID field_id,
                                               Legion::Mapping::PhysicalInstance instance,
                                               InstanceMappingPolicy policy)
{
  FieldMemInfo key{instance.get_tree_id(), field_id, instance.get_location()};

  instance_sets_[std::move(key)].record_instance(
    redop, region, std::move(instance), std::move(policy));
}

void ReductionInstanceManager::erase(const Legion::Mapping::PhysicalInstance& inst)
{
  const auto mem = inst.get_location();
  const auto tid = inst.get_tree_id();

  for (auto fit = instance_sets_.begin(); fit != instance_sets_.end(); /*nothing*/) {
    if ((fit->first.memory != mem) || (fit->first.tid != tid)) {
      ++fit;
      continue;
    }
    if (fit->second.erase(inst)) {
      fit = instance_sets_.erase(fit);
    } else {
      ++fit;
    }
  }
}

/*static*/ ReductionInstanceManager* ReductionInstanceManager::get_instance_manager()
{
  static ReductionInstanceManager manager{};
  return &manager;
}

}  // namespace legate::mapping::detail
