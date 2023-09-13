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

#include "core/mapping/detail/mapping.h"

namespace legate::mapping::detail {

TaskTarget to_target(Processor::Kind kind)
{
  switch (kind) {
    case Processor::Kind::TOC_PROC: return TaskTarget::GPU;
    case Processor::Kind::OMP_PROC: return TaskTarget::OMP;
    case Processor::Kind::LOC_PROC: return TaskTarget::CPU;
    default: LEGATE_ABORT;
  }
  assert(false);
  return TaskTarget::CPU;
}

Processor::Kind to_kind(TaskTarget target)
{
  switch (target) {
    case TaskTarget::GPU: return Processor::Kind::TOC_PROC;
    case TaskTarget::OMP: return Processor::Kind::OMP_PROC;
    case TaskTarget::CPU: return Processor::Kind::LOC_PROC;
    default: LEGATE_ABORT;
  }
  assert(false);
  return Processor::Kind::LOC_PROC;
}

Memory::Kind to_kind(StoreTarget target)
{
  switch (target) {
    case StoreTarget::SYSMEM: return Memory::Kind::SYSTEM_MEM;
    case StoreTarget::FBMEM: return Memory::Kind::GPU_FB_MEM;
    case StoreTarget::ZCMEM: return Memory::Kind::Z_COPY_MEM;
    case StoreTarget::SOCKETMEM: return Memory::Kind::SOCKET_MEM;
    default: LEGATE_ABORT;
  }
  assert(false);
  return Memory::Kind::SYSTEM_MEM;
}

LegateVariantCode to_variant_code(TaskTarget target)
{
  switch (target) {
    case TaskTarget::GPU: return LEGATE_GPU_VARIANT;
    case TaskTarget::OMP: return LEGATE_OMP_VARIANT;
    case TaskTarget::CPU: return LEGATE_CPU_VARIANT;
    default: LEGATE_ABORT;
  }
  assert(false);
  return LEGATE_CPU_VARIANT;
}

void DimOrdering::populate_dimension_ordering(const Store* store,
                                              std::vector<Legion::DimensionKind>& ordering) const
{
  // TODO: We need to implement the relative dimension ordering
  switch (kind) {
    case Kind::C: {
      auto dim = store->region_field().dim();
      for (int32_t idx = dim - 1; idx >= 0; --idx)
        ordering.push_back(static_cast<Legion::DimensionKind>(DIM_X + idx));
      break;
    }
    case Kind::FORTRAN: {
      auto dim = store->region_field().dim();
      for (int32_t idx = 0; idx < dim; ++idx)
        ordering.push_back(static_cast<Legion::DimensionKind>(DIM_X + idx));
      break;
    }
    case Kind::CUSTOM: {
      for (auto idx : dims) ordering.push_back(static_cast<Legion::DimensionKind>(DIM_X + idx));
      break;
    }
  }
}

bool StoreMapping::for_future() const
{
  for (auto& store : stores) return store->is_future();
  return false;
}

bool StoreMapping::for_unbound_store() const
{
  for (auto& store : stores) return store->unbound();
  return false;
}

const Store* StoreMapping::store() const { return stores.front(); }

uint32_t StoreMapping::requirement_index() const
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(stores.size() > 0);
    uint32_t result = -1U;
    for (auto& store : stores) {
      auto idx = store->requirement_index();
      assert(result == -1U || result == idx);
      result = idx;
    }
    return result;
  } else {
    static constexpr uint32_t invalid = -1U;
    if (stores.empty()) return invalid;
    return stores.front()->requirement_index();
  }
}

std::set<uint32_t> StoreMapping::requirement_indices() const
{
  std::set<uint32_t> indices;
  for (auto& store : stores) {
    if (store->is_future()) continue;
    indices.insert(store->region_field().index());
  }
  return indices;
}

std::set<const Legion::RegionRequirement*> StoreMapping::requirements() const
{
  std::set<const Legion::RegionRequirement*> reqs;
  for (auto& store : stores) {
    if (store->is_future()) continue;
    auto* req = store->region_field().get_requirement();
    if (!req->region.exists()) continue;
    reqs.insert(req);
  }
  return reqs;
}

void StoreMapping::populate_layout_constraints(
  Legion::LayoutConstraintSet& layout_constraints) const
{
  std::vector<Legion::DimensionKind> dimension_ordering{};

  if (policy.layout == InstLayout::AOS) dimension_ordering.push_back(DIM_F);
  policy.ordering.impl()->populate_dimension_ordering(stores.front(), dimension_ordering);
  if (policy.layout == InstLayout::SOA) dimension_ordering.push_back(DIM_F);

  layout_constraints.add_constraint(
    Legion::OrderingConstraint(dimension_ordering, false /*contiguous*/));

  layout_constraints.add_constraint(Legion::MemoryConstraint(to_kind(policy.target)));

  std::vector<Legion::FieldID> fields{};
  if (stores.size() > 1) {
    std::set<Legion::FieldID> field_set{};
    for (auto& store : stores) {
      auto field_id = store->region_field().field_id();
      if (field_set.find(field_id) == field_set.end()) {
        fields.push_back(field_id);
        field_set.insert(field_id);
      }
    }
  } else
    fields.push_back(stores.front()->region_field().field_id());
  layout_constraints.add_constraint(
    Legion::FieldConstraint(fields, false /*contiguous*/, false /*inorder*/));
}

/*static*/ std::unique_ptr<StoreMapping> StoreMapping::default_mapping(const Store* store,
                                                                       StoreTarget target,
                                                                       bool exact)
{
  return create(store, InstanceMappingPolicy{}.with_target(target).with_exact(exact));
}

/*static*/ std::unique_ptr<StoreMapping> StoreMapping::create(const Store* store,
                                                              InstanceMappingPolicy&& policy)
{
  auto mapping    = std::make_unique<detail::StoreMapping>();
  mapping->policy = std::move(policy);
  mapping->stores.push_back(store);
  return mapping;
}

}  // namespace legate::mapping::detail
