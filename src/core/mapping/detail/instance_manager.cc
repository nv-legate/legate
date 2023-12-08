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

#include "core/mapping/detail/instance_manager.h"

#include "core/utilities/dispatch.h"
#include "core/utilities/internal_shared_ptr.h"

#include <iostream>
#include <unordered_set>

namespace legate::mapping::detail {

using RegionGroupP = InternalSharedPtr<RegionGroup>;

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

  auto finder = subsumption_cache.find(other);
  if (finder != subsumption_cache.end()) {
    return finder->second;
  }
  for (auto& region : other->regions) {
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

bool InstanceSet::find_instance(Region region,
                                Instance& result,
                                const InstanceMappingPolicy& policy) const
{
  auto finder = groups_.find(region);
  if (finder == groups_.end()) {
    return false;
  }

  auto& group = finder->second;
  if (policy.exact && group->regions.size() > 1) {
    return false;
  }

  auto ifinder = instances_.find(group.get());
  assert(ifinder != instances_.end());

  auto& spec = ifinder->second;
  if (spec.policy.subsumes(policy)) {
    result = spec.instance;
    return true;
  }
  return false;
}

namespace {

// We define "too big" as the size of the "unused" points being bigger than the intersection
bool too_big(size_t union_volume, size_t my_volume, size_t group_volume, size_t intersect_volume)
{
  return (union_volume - (my_volume + group_volume - intersect_volume)) > intersect_volume;
}

}  // namespace

struct construct_overlapping_region_group_fn {
  template <int32_t DIM>
  RegionGroupP operator()(
    const InstanceSet::Region& region,
    const Domain& domain,
    const std::unordered_map<RegionGroup*, InstanceSet::InstanceSpec>& instances)
  {
    auto bound       = domain.bounds<DIM, coord_t>();
    size_t bound_vol = bound.volume();
    std::set<InstanceSet::Region> regions{region};

    if (LegateDefined(LEGATE_USE_DEBUG)) {
      log_instmgr().debug() << "construct_overlapping_region_group( " << region << "," << domain
                            << ")";
    }

    for (const auto& pair : instances) {
      auto& group = pair.first;

      const Rect<DIM> group_bbox = group->bounding_box.bounds<DIM, coord_t>();
      if (LegateDefined(LEGATE_USE_DEBUG)) {
        log_instmgr().debug() << "  check intersection with " << group_bbox;
      }
      auto intersect = bound.intersection(group_bbox);
      if (intersect.empty()) {
        if (LegateDefined(LEGATE_USE_DEBUG)) {
          log_instmgr().debug() << "    no intersection";
        }
        continue;
      }

      // Only allow merging if the bloating isn't "too big"
      auto union_bbox            = bound.union_bbox(group_bbox);
      const size_t union_vol     = union_bbox.volume();
      const size_t group_vol     = group_bbox.volume();
      const size_t intersect_vol = intersect.volume();
      if (too_big(union_vol, bound_vol, group_vol, intersect_vol)) {
        if (LegateDefined(LEGATE_USE_DEBUG)) {
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
      if (LegateDefined(LEGATE_USE_DEBUG)) {
        log_instmgr().debug() << "    bounds updated: " << bound << " ~> " << union_bbox;
      }

      bound     = union_bbox;
      bound_vol = union_vol;
    }

    return make_internal_shared<RegionGroup>(std::move(regions), Domain(bound));
  }
};

RegionGroupP InstanceSet::construct_overlapping_region_group(const Region& region,
                                                             const Domain& domain,
                                                             bool exact) const
{
  auto finder = groups_.find(region);
  if (finder == groups_.end()) {
    return dim_dispatch(
      domain.get_dim(), construct_overlapping_region_group_fn{}, region, domain, instances_);
  }

  if (!exact || finder->second->regions.size() == 1) {
    return finder->second;
  }
  return make_internal_shared<RegionGroup>(std::set<Region>{region}, domain);
}

std::set<InstanceSet::Instance> InstanceSet::record_instance(const RegionGroupP& group,
                                                             const Instance& instance,
                                                             const InstanceMappingPolicy& policy)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
#ifdef DEBUG_INSTANCE_MANAGER
    log_instmgr().debug() << "===== before adding an entry " << *group << " ~> " << instance
                          << " =====";
#endif
  }
  dump_and_sanity_check();

  std::set<Instance> replaced;
  std::set<RegionGroupP> removed_groups;

  auto finder = instances_.find(group.get());
  if (finder != instances_.end()) {
    replaced.insert(finder->second.instance);
    finder->second = InstanceSpec{instance, policy};
  } else {
    instances_[group.get()] = InstanceSpec{instance, policy};
  }

  for (auto& region : group->regions) {
    auto it = groups_.find(region);
    if (it == groups_.end()) {
      groups_[region] = group;
    } else if (it->second != group) {
      removed_groups.insert(it->second);
      it->second = group;
    }
  }

  for (auto& removed_group : removed_groups) {
    // Because of exact policies, we can't simply remove the groups where regions in the `group`
    // originally belonged, because one region can be included in multiple region groups. (Note that
    // the exact mapping bypasses the coalescing heuristic and always creates a fresh singleton
    // group.) So, before we prune out each of those potentially obsolete groups, we need to
    // make sure that it has no remaining references.
    bool can_remove = true;
    for (const Region& rg : removed_group->regions) {
      if (groups_.at(rg) == removed_group) {
        can_remove = false;
        break;
      }
    }
    if (can_remove) {
      auto it = instances_.find(removed_group.get());
      replaced.insert(it->second.instance);
      instances_.erase(it);
    }
  }

  replaced.erase(instance);

  if (LegateDefined(LEGATE_USE_DEBUG)) {
#ifdef DEBUG_INSTANCE_MANAGER
    log_instmgr().debug() << "===== after adding an entry " << *group << " ~> " << instance
                          << " =====";
#endif
    dump_and_sanity_check();
  }
  return replaced;
}

bool InstanceSet::erase(const Instance& inst)
{
  std::set<RegionGroup*> filtered_groups;
  if (LegateDefined(LEGATE_USE_DEBUG)) {
#ifdef DEBUG_INSTANCE_MANAGER
    log_instmgr().debug() << "===== before erasing an instance " << inst << " =====";
#endif
  }
  dump_and_sanity_check();

  for (auto it = instances_.begin(); it != instances_.end(); /*nothing*/) {
    if (it->second.instance == inst) {
      auto to_erase = it++;
      filtered_groups.insert(to_erase->first);
      instances_.erase(to_erase);
    } else {
      ++it;
    }
  }

  std::set<Region> filtered_regions;
  for (RegionGroup* group : filtered_groups) {
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

  if (LegateDefined(LEGATE_USE_DEBUG)) {
#ifdef DEBUG_INSTANCE_MANAGER
    log_instmgr().debug() << "===== after erasing an instance " << inst << " =====";
#endif
  }
  dump_and_sanity_check();

  return instances_.empty();
}

size_t InstanceSet::get_instance_size() const
{
  size_t sum = 0;
  for (auto& pair : instances_) {
    sum += pair.second.instance.get_instance_size();
  }
  return sum;
}

void InstanceSet::dump_and_sanity_check() const
{
#ifdef DEBUG_INSTANCE_MANAGER
  for (auto& entry : groups_) {
    log_instmgr().debug() << "  " << entry.first << " ~> " << *entry.second;
  }
  for (auto& entry : instances_) {
    log_instmgr().debug() << "  " << *entry.first << " ~> " << entry.second.instance;
  }
#endif
  std::unordered_set<RegionGroup*> found_groups;
  for (auto& entry : groups_) {
    found_groups.insert(entry.second.get());
    assert(instances_.count(entry.second.get()) > 0);
    assert(entry.second->regions.count(entry.first) > 0);
  }
  for (auto& entry : instances_) {
    assert(found_groups.count(entry.first) > 0);
  }
}

bool ReductionInstanceSet::find_instance(ReductionOpID& redop,
                                         Region& region,
                                         Instance& result,
                                         const InstanceMappingPolicy& policy) const
{
  auto finder = instances_.find(region);
  if (finder == instances_.end()) {
    return false;
  }
  auto& spec = finder->second;
  if (spec.policy == policy && spec.redop == redop) {
    result = spec.instance;
    return true;
  }
  return false;
}

void ReductionInstanceSet::record_instance(ReductionOpID& redop,
                                           Region& region,
                                           Instance& instance,
                                           const InstanceMappingPolicy& policy)
{
  auto finder = instances_.find(region);
  if (finder != instances_.end()) {
    auto& spec = finder->second;
    if (spec.policy != policy || spec.redop != redop) {
      instances_.insert_or_assign(region, ReductionInstanceSpec{redop, instance, policy});
    }
  } else {
    instances_[region] = {redop, instance, policy};
  }
}

bool ReductionInstanceSet::erase(const Instance& inst)
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

bool InstanceManager::find_instance(Region region,
                                    FieldID field_id,
                                    Memory memory,
                                    Instance& result,
                                    const InstanceMappingPolicy& policy)
{
  auto finder = instance_sets_.find({region.get_tree_id(), field_id, memory});
  return policy.allocation != AllocPolicy::MUST_ALLOC && finder != instance_sets_.end() &&
         finder->second.find_instance(region, result, policy);
}

RegionGroupP InstanceManager::find_region_group(const Region& region,
                                                const Domain& domain,
                                                FieldID field_id,
                                                Memory memory,
                                                bool exact /*=false*/)
{
  RegionGroupP result{};

  auto finder = instance_sets_.find({region.get_tree_id(), field_id, memory});
  if (finder == instance_sets_.end() || exact) {
    result = make_internal_shared<RegionGroup>(std::set<Region>{region}, domain);
  } else {
    result = finder->second.construct_overlapping_region_group(region, domain, exact);
  }

  if (LegateDefined(LEGATE_USE_DEBUG)) {
    log_instmgr().debug() << "find_region_group(" << region << "," << domain << "," << field_id
                          << "," << memory << "," << exact << ") ~> " << *result;
  }

  return result;
}

std::set<InstanceManager::Instance> InstanceManager::record_instance(
  const RegionGroupP& group,
  FieldID field_id,
  const Instance& instance,
  const InstanceMappingPolicy& policy)
{
  FieldMemInfo key{instance.get_tree_id(), field_id, instance.get_location()};
  return instance_sets_[std::move(key)].record_instance(group, instance, policy);
}

void InstanceManager::erase(const Instance& inst)
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

std::map<Memory, size_t> InstanceManager::aggregate_instance_sizes() const
{
  std::map<Memory, size_t> result;

  for (auto& pair : instance_sets_) {
    auto& memory = pair.first.memory;
    if (result.find(memory) == result.end()) {
      result[memory] = 0;
    }
    result[memory] += pair.second.get_instance_size();
  }
  return result;
}

/*static*/ InstanceManager* InstanceManager::get_instance_manager()
{
  static InstanceManager* manager{nullptr};

  if (nullptr == manager) {
    manager = new InstanceManager{};
  }
  return manager;
}

bool ReductionInstanceManager::find_instance(ReductionOpID& redop,
                                             Region region,
                                             FieldID field_id,
                                             Memory memory,
                                             Instance& result,
                                             const InstanceMappingPolicy& policy)
{
  auto finder = instance_sets_.find({region.get_tree_id(), field_id, memory});
  return policy.allocation != AllocPolicy::MUST_ALLOC && finder != instance_sets_.end() &&
         finder->second.find_instance(redop, region, result, policy);
}

void ReductionInstanceManager::record_instance(ReductionOpID& redop,
                                               Region region,
                                               FieldID field_id,
                                               Instance instance,
                                               const InstanceMappingPolicy& policy)
{
  FieldMemInfo key{instance.get_tree_id(), field_id, instance.get_location()};
  instance_sets_[std::move(key)].record_instance(redop, region, instance, policy);
}

void ReductionInstanceManager::erase(const Instance& inst)
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
  static ReductionInstanceManager* manager{nullptr};

  if (manager == nullptr) {
    manager = new ReductionInstanceManager{};
  }
  return manager;
}

}  // namespace legate::mapping::detail
