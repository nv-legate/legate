/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/instance_manager.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/dispatch.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <algorithm>
#include <iostream>
#include <unordered_set>

namespace legate::mapping::detail {

namespace {

[[nodiscard]] Legion::Logger& log_instmgr()
{
  static Legion::Logger log{"legate.instmgr"};

  return log;
}

}  // namespace

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
  const Legion::Mapping::MapperContext& ctx,
  const Legion::Mapping::MapperRuntime* runtime,
  const Legion::LogicalRegion& region,
  const InstanceMappingPolicy& policy,
  const Legion::LayoutConstraintSet& layout_constraints) const
{
  // Helper function to get a group's instance. Non-nullopt if the instance satisfies the query.
  const auto get_valid_instance = [&](const InternalSharedPtr<RegionGroup>& group)
    -> std::optional<Legion::Mapping::PhysicalInstance> {
    const auto ifinder = instances_.find(group.get());

    LEGATE_CHECK(ifinder != instances_.end());
    if (auto&& spec = ifinder->second;
        (spec.policy.exact || !policy.exact) && spec.instance.entails(layout_constraints)) {
      return spec.instance;
    }
    return std::nullopt;
  };

  // Prioritize searching for the exact region in the cache first.
  const auto it = groups_.find(region);

  if (it != groups_.end()) {
    if (const auto instance = get_valid_instance(it->second); instance.has_value()) {
      return instance;
    }
  }

  if (policy.exact) {
    // The logic below will search for an instance in which the domain
    // of the region query is a subset of the domain of the instance. However,
    // the logic below doesn't guarantee that the two domains will be equal
    // (i.e. exact in Legate terminology). As a result, if the query expects
    // an exact instance, then simply return a cache miss because the below
    // logic doesn't guarantee it.
    return std::nullopt;
  }

  // Search for a group in the instance set that covers the region's domain and
  // that has a valid instance which satisfies the query.
  std::vector<Legion::Domain> region_domains;

  runtime->get_index_space_domains(ctx, region.get_index_space(), region_domains);

  auto groups_iter = groups_.begin();

  while (groups_iter != groups_.end()) {
    // Consider group's instance if the region's domain subsets the group's domain.
    const auto valid_group_iter = std::find_if(
      groups_iter,
      groups_.end(),
      [&](const std::pair<Legion::LogicalRegion, InternalSharedPtr<RegionGroup>>& pair) {
        const Domain& group_domain = pair.second->bounding_box;

        return std::all_of(region_domains.begin(), region_domains.end(), [&](auto&& domain) {
          return group_domain.contains(domain.lo()) && group_domain.contains(domain.hi());
        });
      });

    if (valid_group_iter == groups_.end()) {
      // No more satisfying groups, stop searching.
      break;
    }

    if (const auto instance = get_valid_instance(valid_group_iter->second); instance.has_value()) {
      return instance;
    }

    // Instance for the matching group is invalid, keep searching.
    groups_iter = std::next(valid_group_iter);
  }

  // No valid instances, cache miss.
  return std::nullopt;
}

namespace {

// We define "too big" as the size of the "unused" points being bigger than the intersection
[[nodiscard]] bool bloat_too_big(std::size_t union_volume,
                                 std::size_t my_volume,
                                 std::size_t group_volume,
                                 std::size_t intersect_volume)
{
  return (union_volume - (my_volume + group_volume - intersect_volume)) > intersect_volume;
}

// When the two rects can be combined, the last two arguments will be updated with a union of the
// two box and the total volume, respectively
template <std::int32_t DIM>
[[nodiscard]] bool can_combine(const Rect<DIM>& next,
                               Rect<DIM>* current,
                               std::size_t* current_volume)
{
  const auto intersect = current->intersection(next);
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_instmgr().debug() << "  check intersection with " << next;
  }
  if (intersect.empty()) {
    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      log_instmgr().debug() << "    no intersection";
    }
    return false;
  }

  // Only allow merging if the bloating isn't "too big"
  const auto union_bbox       = current->union_bbox(next);
  const auto union_volume     = union_bbox.volume();
  const auto next_volume      = next.volume();
  const auto intersect_volume = intersect.volume();
  if (bloat_too_big(union_volume, *current_volume, next_volume, intersect_volume)) {
    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      log_instmgr().debug() << "    too big a bloat to merge (current:" << *current
                            << ",next:" << next << ",union:" << union_bbox
                            << ",current_volume:" << *current_volume
                            << ",union_volume:" << union_volume
                            << ",intersect_volume:" << intersect_volume << ")";
    }
    return false;
  }

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_instmgr().debug() << "    bbox updated: " << *current << " ~> " << union_bbox;
  }

  *current        = union_bbox;
  *current_volume = union_volume;
  return true;
}

[[nodiscard]] InternalSharedPtr<RegionGroup> create_initial_region_group(
  Legion::LogicalRegion region, const Domain& domain)
{
  // TODO(wonchanl): a slice prefetching logic will be added here
  return make_internal_shared<RegionGroup>(std::set<Legion::LogicalRegion>{std::move(region)},
                                           domain);
}

class ConstructOverlappingRegionGroupFn {
 public:
  template <std::int32_t DIM>
  [[nodiscard]] InternalSharedPtr<RegionGroup> operator()(
    InternalSharedPtr<RegionGroup> group,
    const std::unordered_map<RegionGroup*, InstanceSet::InstanceSpec>& instances,
    const std::unordered_map<InternalSharedPtr<RegionGroup>, std::uint64_t>& pending_instances)
    const
  {
    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      log_instmgr().debug() << " construct_overlapping_region_group( " << *group << ")";
    }

    auto bbox     = group->bounding_box.template bounds<DIM, coord_t>();
    auto bbox_vol = bbox.volume();
    std::vector<RegionGroup*> to_combine{};

    const auto combine_new_region_group = [&](auto&& new_group) {
      // Avoid self-comparison
      if (new_group == group.get()) {
        return;
      }
      if (can_combine(new_group->bounding_box.template bounds<DIM, coord_t>(), &bbox, &bbox_vol)) {
        to_combine.push_back(new_group);
      }
    };

    // Find all the overlapping groups that are worth combining with the current group
    for (auto&& [next_group, _] : instances) {
      combine_new_region_group(next_group);
    }

    for (auto&& [next_group, _] : pending_instances) {
      combine_new_region_group(next_group.get());
    }

    if (to_combine.empty()) {
      if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
        log_instmgr().debug() << " ~> " << *group << " (nothing to combine)";
      }
      return group;
    }

    // Copy is intentional, as we're about to create a new group
    auto all_regions = group->regions;
    for (auto&& next_group : to_combine) {
      // NOTE: It is critical that we maintain the invariant that if at least one region is mapped
      // to a group in the instances_ table, that group is still present on the groups_ table, and
      // thus there's at least one shared_ptr remaining that points to it. Otherwise we run the risk
      // that a group pointer stored on the instances_ table points to a group that's been collected
      all_regions.insert(next_group->regions.begin(), next_group->regions.end());
    }

    // If there's no change to the set of regions, then we deduplicate the group
    if (all_regions.size() == group->regions.size()) {
      if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
        log_instmgr().debug() << " ~> " << *group << " (deduplicated)";
      }
      return group;
    }
    for (auto&& next_group : to_combine) {
      if (all_regions.size() == next_group->regions.size()) {
        if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
          log_instmgr().debug() << " ~> " << *next_group << " (deduplicated)";
        }
        return next_group->shared_from_this();
      }
    }

    auto new_group = make_internal_shared<RegionGroup>(std::move(all_regions), Domain{bbox});
    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      log_instmgr().debug() << " ~> " << *new_group << " (created)";
    }
    return new_group;
  }
};

}  // namespace

InternalSharedPtr<RegionGroup> InstanceSet::find_or_create_region_group(
  const Legion::LogicalRegion& region, const Domain& domain, bool exact) const
{
  auto finder = groups_.find(region);

  // Handle the exact case separately to make the code clear
  if (exact) {
    if (finder != groups_.end() && finder->second->regions.size() == 1) {
      return finder->second;
    }
    return make_internal_shared<RegionGroup>(std::set<Legion::LogicalRegion>{region}, domain);
  }

  auto group =
    finder != groups_.end() ? finder->second : create_initial_region_group(region, domain);
  return dim_dispatch(domain.get_dim(),
                      ConstructOverlappingRegionGroupFn{},
                      std::move(group),
                      instances_,
                      pending_instances_);
}

void InstanceSet::record_instance(const Legion::LogicalRegion& region,
                                  const InternalSharedPtr<RegionGroup>& group,
                                  Legion::Mapping::PhysicalInstance instance,
                                  InstanceMappingPolicy policy)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_instmgr().debug() << "===== before record_instance(" << region << ", " << *group << ",  "
                          << instance << ") =====";
    dump_and_sanity_check_();
  }

  // If this is the last pending request of a region group, we should remove it from the map
  remove_pending_instance(group);

  const auto& inst =
    instances_.insert_or_assign(group.get(), InstanceSpec{std::move(instance), std::move(policy)})
      .first->second.instance;

  // Use of InternalSharedPtr vs raw RegionGroup * is deliberate. We swap the group down below,
  // and if the old region group is the last one left, we should delete it until we can remove
  // it from our instances_...
  std::unordered_set<InternalSharedPtr<RegionGroup>> removed_groups;

  for (auto&& rgn : group->regions) {
    const auto [it, inserted] = groups_.try_emplace(rgn, group);
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
    log_instmgr().debug() << "===== after record_instance(" << region << ", " << *group << ",  "
                          << inst << ") =====";
    dump_and_sanity_check_();
  }
}

void InstanceSet::record_pending_instance_creation(InternalSharedPtr<RegionGroup> group)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_instmgr().debug() << "===== before record_pending_instance_creation(" << *group
                          << ") =====";
    dump_and_sanity_check_();
  }

  // Increment the pending instance counter
  auto&& [grp, counter] = *pending_instances_.try_emplace(std::move(group)).first;
  ++counter;

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_instmgr().debug() << "===== after record_pending_instance_creation(" << *grp << ") =====";
    dump_and_sanity_check_();
  }
}

void InstanceSet::remove_pending_instance(const InternalSharedPtr<RegionGroup>& group)
{
  if (const auto it = pending_instances_.find(group); --it->second == 0) {
    pending_instances_.erase(it);
  }
}

bool InstanceSet::erase(const Legion::Mapping::PhysicalInstance& inst)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_instmgr().debug() << "===== before erasing an instance " << inst << " =====";
    dump_and_sanity_check_();
  }

  std::set<RegionGroup*> filtered_groups;
  auto did_erase = false;

  for (auto it = instances_.begin(); it != instances_.end(); /*nothing*/) {
    if (it->second.instance == inst) {
      filtered_groups.insert(it->first);
      it        = instances_.erase(it);
      did_erase = true;
    } else {
      ++it;
    }
  }

  std::set<Legion::LogicalRegion> filtered_regions;
  for (const RegionGroup* group : filtered_groups) {
    // We have to do this in two steps; we don't want to remove the last shared_ptr to a group
    // while iterating over the same group's regions
    std::copy_if(group->regions.cbegin(),
                 group->regions.cend(),
                 std::inserter(filtered_regions, filtered_regions.begin()),
                 [&](const Legion::LogicalRegion& region) {
                   const auto finder = groups_.find(region);

                   return finder != groups_.end() && finder->second.get() == group;
                 });
  }

  for (auto&& region : filtered_regions) {
    groups_.erase(region);
  }

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_instmgr().debug() << "===== after erasing an instance " << inst << " =====";
    dump_and_sanity_check_();
  }

  return did_erase;
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
  log_instmgr().debug() << " REGION GROUPS";
  for (auto&& [region, group] : groups_) {
    log_instmgr().debug() << "  " << region << " ~> " << *group;
  }
  log_instmgr().debug() << " INSTANCES";
  for (auto&& [group, instance_spec] : instances_) {
    log_instmgr().debug() << "  " << *group << " ~> " << instance_spec.instance;
  }
  log_instmgr().debug() << " PENDING INSTANCES";
  for (auto&& [group, count] : pending_instances_) {
    log_instmgr().debug() << "  " << *group << " ~> " << count;
  }

  std::unordered_set<RegionGroup*> found_groups;
  for (auto&& [region, group] : groups_) {
    LEGATE_CHECK(group->regions.count(region) > 0);
    LEGATE_CHECK(instances_.count(group.get()) > 0);
    found_groups.insert(group.get());
  }
  for (auto&& [group, _] : instances_) {
    if (found_groups.count(group) == 0) {
      log_instmgr().debug() << "  " << *group << " is dangling";
      LEGATE_CHECK(false);
    }
  }
}

std::optional<Legion::Mapping::PhysicalInstance> ReductionInstanceSet::find_instance(
  GlobalRedopID redop,
  const Legion::LogicalRegion& region,
  const Legion::LayoutConstraintSet& layout_constraints) const
{
  if (const auto it = instances_.find(region); it != instances_.end()) {
    // TODO(mpapadakis): The ReductionInstanceManager does not currently respect requests for
    // "exact" instances, so ignore that field when polling the cache. If this is updated here, also
    // set tight_bounds to match, in BaseMapper::map_reduction_instance_::find_instance.
    if (auto&& spec = it->second;
        spec.redop == redop && spec.instance.entails(layout_constraints)) {
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
  auto did_erase = false;

  for (auto it = instances_.begin(); it != instances_.end(); /*nothing*/) {
    if (it->second.instance == inst) {
      it        = instances_.erase(it);
      did_erase = true;
    } else {
      ++it;
    }
  }
  return did_erase;
}

// ==========================================================================================

template <typename T>
bool BaseInstanceManager::do_erase_(
  std::unordered_map<FieldMemInfo, T, hasher<FieldMemInfo>>* instance_sets,
  const Legion::Mapping::PhysicalInstance& inst)
{
  const auto mem = inst.get_location();
  const auto tid = inst.get_tree_id();
  auto did_erase = false;

  for (auto fit = instance_sets->begin(); fit != instance_sets->end(); /*nothing*/) {
    if ((fit->first.memory != mem) || (fit->first.tid != tid)) {
      ++fit;
      continue;
    }

    T& sub_inst = fit->second;

    if (sub_inst.erase(inst)) {
      did_erase = true;
    }

    if (sub_inst.empty()) {
      fit = instance_sets->erase(fit);
    } else {
      ++fit;
    }
  }
  return did_erase;
}

// ==========================================================================================

std::optional<Legion::Mapping::PhysicalInstance> InstanceManager::find_instance(
  const Legion::Mapping::MapperContext& ctx,
  const Legion::Mapping::MapperRuntime* runtime,
  const Legion::LogicalRegion& region,
  Legion::FieldID field_id,
  Memory memory,
  const InstanceMappingPolicy& policy,
  const Legion::LayoutConstraintSet& layout_constraints)
{
  // MUST_ALLOC should never reach here
  LEGATE_ASSERT(policy.allocation != AllocPolicy::MUST_ALLOC);

  const auto it = instance_sets_.find({region.get_tree_id(), field_id, memory});

  if (it == instance_sets_.end()) {
    return std::nullopt;
  }

  return it->second.find_instance(ctx, runtime, region, policy, layout_constraints);
}

InternalSharedPtr<RegionGroup> InstanceManager::find_region_group(
  const Legion::LogicalRegion& region,
  const Domain& domain,
  Legion::FieldID field_id,
  Memory memory,
  bool exact /*=false*/)
{
  InternalSharedPtr<RegionGroup> result = [&] {
    auto&& [_, instance_set] =
      *instance_sets_.try_emplace({region.get_tree_id(), field_id, memory}).first;

    auto group = instance_set.find_or_create_region_group(region, domain, exact);

    // When the whole cached instance creation was done atomically, we had a nice property that
    // region groups are monotonically increasing. For example, if we have regions R, S1, and S2,
    // and they all overlap with each other to a degree that `find_or_create_region_group` would put
    // them in the same group, in the old days, the progression of region groups for these regions
    // would have been either {R, S1} -> {R, S1, S2} or {R, S2} -> {R, S2, S1}. In the new world
    // where the cache retrieval, instance creation, and cache update are performed separately, R,
    // S1, and S2 would no longer go through the same progression, but they can form two overlapping
    // region groups {R, S1} and {R, S2}.  This is problematic because each group will get mapped to
    // a separate instance and the two instances will need to constantly fetch updates from the
    // other for the overlapping portion.  We can maintain monotonicity by recording the newly
    // region group early here.
    instance_set.record_pending_instance_creation(group);
    return group;
  }();

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_instmgr().debug() << " find_region_group(" << region << "," << domain << "," << field_id
                          << "," << memory << "," << exact << ") ~> " << *result;
  }

  return result;
}

void InstanceManager::record_instance(const Legion::LogicalRegion& region,
                                      const InternalSharedPtr<RegionGroup>& group,
                                      Legion::FieldID field_id,
                                      Legion::Mapping::PhysicalInstance instance,
                                      InstanceMappingPolicy policy)
{
  FieldMemInfo key{instance.get_tree_id(), field_id, instance.get_location()};

  instance_sets_[std::move(key)].record_instance(
    region, group, std::move(instance), std::move(policy));
}

void InstanceManager::remove_pending_instance(const Legion::LogicalRegion& region,
                                              const InternalSharedPtr<RegionGroup>& group,
                                              Legion::FieldID field_id,
                                              const Memory& memory)
{
  auto finder = instance_sets_.find(FieldMemInfo{region.get_tree_id(), field_id, memory});
  LEGATE_ASSERT(finder != instance_sets_.end());
  auto&& instance_set = finder->second;
  instance_set.remove_pending_instance(group);
  if (instance_set.empty()) {
    instance_sets_.erase(finder);
  }
}

bool InstanceManager::erase(const Legion::Mapping::PhysicalInstance& inst)
{
  return do_erase_(&instance_sets_, inst);
}

std::map<Memory, std::size_t> InstanceManager::aggregate_instance_sizes() const
{
  std::map<Memory, std::size_t> result;

  for (auto&& [field_mem_info, instance_set] : instance_sets_) {
    result[field_mem_info.memory] += instance_set.get_instance_size();
  }
  return result;
}

std::optional<Legion::Mapping::PhysicalInstance> ReductionInstanceManager::find_instance(
  GlobalRedopID redop,
  const Legion::LogicalRegion& region,
  Legion::FieldID field_id,
  Memory memory,
  const InstanceMappingPolicy& policy,
  const Legion::LayoutConstraintSet& layout_constraints)
{
  // MUST_ALLOC should never reach here
  LEGATE_ASSERT(policy.allocation != AllocPolicy::MUST_ALLOC);

  const auto it = instance_sets_.find({region.get_tree_id(), field_id, memory});

  if (it == instance_sets_.end()) {
    return std::nullopt;
  }

  return it->second.find_instance(redop, region, layout_constraints);
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

bool ReductionInstanceManager::erase(const Legion::Mapping::PhysicalInstance& inst)
{
  return do_erase_(&instance_sets_, inst);
}

}  // namespace legate::mapping::detail
