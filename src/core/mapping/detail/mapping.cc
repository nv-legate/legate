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

#include "core/mapping/detail/mapping.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_set>

namespace legate::mapping::detail {

TaskTarget to_target(Processor::Kind kind)
{
  switch (kind) {
    case Processor::Kind::TOC_PROC: return TaskTarget::GPU;
    case Processor::Kind::OMP_PROC: return TaskTarget::OMP;
    case Processor::Kind::LOC_PROC: return TaskTarget::CPU;
    default: LEGATE_ABORT("Unhandled Processor::Kind ", traits::detail::to_underlying(kind));
  }
  return TaskTarget::CPU;
}

StoreTarget to_target(Memory::Kind kind)
{
  switch (kind) {
    case Memory::Kind::SYSTEM_MEM: return StoreTarget::SYSMEM;
    case Memory::Kind::GPU_FB_MEM: return StoreTarget::FBMEM;
    case Memory::Kind::Z_COPY_MEM: return StoreTarget::ZCMEM;
    case Memory::Kind::SOCKET_MEM: return StoreTarget::SOCKETMEM;
    default: LEGATE_ABORT("Unhandled Processor::Kind ", traits::detail::to_underlying(kind));
  }
  return StoreTarget::SYSMEM;
}

Processor::Kind to_kind(TaskTarget target)
{
  switch (target) {
    case TaskTarget::GPU: return Processor::Kind::TOC_PROC;
    case TaskTarget::OMP: return Processor::Kind::OMP_PROC;
    case TaskTarget::CPU: return Processor::Kind::LOC_PROC;
  }
  return Processor::Kind::LOC_PROC;
}

Memory::Kind to_kind(StoreTarget target)
{
  switch (target) {
    case StoreTarget::SYSMEM: return Memory::Kind::SYSTEM_MEM;
    case StoreTarget::FBMEM: return Memory::Kind::GPU_FB_MEM;
    case StoreTarget::ZCMEM: return Memory::Kind::Z_COPY_MEM;
    case StoreTarget::SOCKETMEM: return Memory::Kind::SOCKET_MEM;
  }
  return Memory::Kind::SYSTEM_MEM;
}

VariantCode to_variant_code(TaskTarget target)
{
  switch (target) {
    case TaskTarget::GPU: return VariantCode::GPU;
    case TaskTarget::OMP: return VariantCode::OMP;
    case TaskTarget::CPU: return VariantCode::CPU;
  }
  return VariantCode::CPU;
}

VariantCode to_variant_code(Processor::Kind kind) { return to_variant_code(to_target(kind)); }

void DimOrdering::populate_dimension_ordering(std::uint32_t ndim,
                                              std::vector<Legion::DimensionKind>& ordering) const
{
  // TODO(wonchanl): We need to implement the relative dimension ordering
  switch (kind) {
    case Kind::C: {
      LEGATE_ASSERT(ndim > 0);
      ordering.reserve(ordering.size() + ndim);
      for (auto dim = static_cast<std::int32_t>(ndim) - 1; dim >= 0; --dim) {
        ordering.push_back(static_cast<Legion::DimensionKind>(LEGION_DIM_X + dim));
      }
      break;
    }
    case Kind::FORTRAN: {
      ordering.reserve(ordering.size() + ndim);
      for (std::uint32_t dim = 0; dim < ndim; ++dim) {
        ordering.push_back(static_cast<Legion::DimensionKind>(LEGION_DIM_X + dim));
      }
      break;
    }
    case Kind::CUSTOM: {
      ordering.reserve(ordering.size() + dims.size());
      for (auto dim : dims) {
        ordering.push_back(static_cast<Legion::DimensionKind>(LEGION_DIM_X + dim));
      }
      break;
    }
  }
}

bool StoreMapping::for_future() const
{
  return std::any_of(
    stores.begin(), stores.end(), [](const Store* store) { return store->is_future(); });
}

bool StoreMapping::for_unbound_store() const
{
  return std::any_of(
    stores.begin(), stores.end(), [](const Store* store) { return store->unbound(); });
}

const Store* StoreMapping::store() const { return stores.front(); }

std::uint32_t StoreMapping::requirement_index() const
{
  static constexpr std::uint32_t INVALID = -1U;

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    std::uint32_t result = INVALID;

    LEGATE_ASSERT(!stores.empty());
    for (const auto* store : stores) {
      const auto idx = store->requirement_index();

      LEGATE_ASSERT(result == INVALID || result == idx);
      result = idx;
    }
    return result;
  }

  if (stores.empty()) {
    return INVALID;
  }
  return store()->requirement_index();
}

std::set<std::uint32_t> StoreMapping::requirement_indices() const
{
  std::set<std::uint32_t> indices;

  for (auto&& store : stores) {
    if (!store->is_future()) {
      indices.insert(store->region_field().index());
    }
  }
  return indices;
}

std::set<const Legion::RegionRequirement*> StoreMapping::requirements() const
{
  std::set<const Legion::RegionRequirement*> reqs;

  for (auto&& store : stores) {
    if (store->is_future()) {
      continue;
    }

    if (const auto* req = store->region_field().get_requirement(); req->region.exists()) {
      reqs.insert(req);
    }
  }
  return reqs;
}

void StoreMapping::populate_layout_constraints(
  Legion::LayoutConstraintSet& layout_constraints) const
{
  auto&& first_region_field = store()->region_field();
  std::vector<Legion::DimensionKind> dimension_ordering{};

  dimension_ordering.reserve(
    (policy.layout == InstLayout::AOS || policy.layout == InstLayout::SOA) +
    static_cast<std::size_t>(first_region_field.dim()));
  if (policy.layout == InstLayout::AOS) {
    dimension_ordering.push_back(LEGION_DIM_F);
  }
  policy.ordering.impl()->populate_dimension_ordering(
    static_cast<std::uint32_t>(first_region_field.dim()), dimension_ordering);
  if (policy.layout == InstLayout::SOA) {
    dimension_ordering.push_back(LEGION_DIM_F);
  }
  // This 2-step is necessary because Legion::OrderingConstraint constructor takes the vector
  // by const-ref. Similarly, layout_constraints.add_constraint() ALSO takes its
  // OrderingConstraint by const-ref, meaning that if we go the usual route of
  //
  // layout_constraints.add_constraint(Legion::OrderingConstraint{dimension_ordering, false})
  //
  // ...would result in not 1, not 2, but 3 deep copies of our vector! Now it's a 0-copy
  // operation (a move)
  layout_constraints.ordering_constraint.ordering   = std::move(dimension_ordering);
  layout_constraints.ordering_constraint.contiguous = false;
  layout_constraints.add_constraint(Legion::MemoryConstraint{to_kind(policy.target)});

  std::vector<Legion::FieldID> fields{};

  if (stores.size() > 1) {
    std::unordered_set<Legion::FieldID> field_set{};

    fields.reserve(stores.size());
    field_set.reserve(stores.size());
    for (auto&& store : stores) {
      const auto field_id      = store->region_field().field_id();
      const auto [_, inserted] = field_set.emplace(field_id);

      if (inserted) {
        fields.push_back(field_id);
      }
    }
  } else {
    fields.push_back(first_region_field.field_id());
  }
  layout_constraints.field_constraint.field_set  = std::move(fields);
  layout_constraints.field_constraint.contiguous = false;
  layout_constraints.field_constraint.inorder    = false;
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
  auto mapping = std::make_unique<detail::StoreMapping>();

  mapping->policy = std::move(policy);
  mapping->stores.push_back(store);
  return mapping;
}

}  // namespace legate::mapping::detail
