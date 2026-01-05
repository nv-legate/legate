/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/mapping.h>

#include <legate/utilities/detail/type_traits.h>

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
    case Processor::Kind::LOC_PROC:
      [[fallthrough]];
      // Note, returning TaskTarget::CPU for Processor::Kind::PY_PROC isn't technically
      // correct. This is "safe" to do because the only caller that might pass PY_PROC is the
      // inline task launch.
    case Processor::Kind::PY_PROC: return TaskTarget::CPU;
    case Processor::Kind::NO_KIND: [[fallthrough]];
    case Processor::Kind::UTIL_PROC: [[fallthrough]];
    case Processor::Kind::IO_PROC: [[fallthrough]];
    case Processor::Kind::PROC_GROUP: [[fallthrough]];
    case Processor::Kind::PROC_SET: break;
  }
  LEGATE_ABORT("Unhandled Processor::Kind ", legate::detail::to_underlying(kind));
}

TaskTarget get_matching_task_target(StoreTarget target)
{
  switch (target) {
    case StoreTarget::ZCMEM: [[fallthrough]];
    case StoreTarget::FBMEM: return TaskTarget::GPU;
    case StoreTarget::SOCKETMEM: return TaskTarget::OMP;
    case StoreTarget::SYSMEM: return TaskTarget::CPU;
  }
  LEGATE_ABORT("Unhandled StoreTarget: ", target);
}

StoreTarget to_target(Memory::Kind kind)
{
  switch (kind) {
    case Memory::Kind::SYSTEM_MEM: return StoreTarget::SYSMEM;
    case Memory::Kind::GPU_FB_MEM: return StoreTarget::FBMEM;
    case Memory::Kind::Z_COPY_MEM: return StoreTarget::ZCMEM;
    case Memory::Kind::SOCKET_MEM: return StoreTarget::SOCKETMEM;
    case Memory::Kind::NO_MEMKIND: [[fallthrough]];
    case Memory::Kind::GLOBAL_MEM: [[fallthrough]];
    case Memory::Kind::REGDMA_MEM: [[fallthrough]];
    case Memory::Kind::DISK_MEM: [[fallthrough]];
    case Memory::Kind::HDF_MEM: [[fallthrough]];
    case Memory::Kind::FILE_MEM: [[fallthrough]];
    case Memory::Kind::LEVEL3_CACHE: [[fallthrough]];
    case Memory::Kind::LEVEL2_CACHE: [[fallthrough]];
    case Memory::Kind::LEVEL1_CACHE: [[fallthrough]];
    case Memory::Kind::GPU_MANAGED_MEM: [[fallthrough]];
    case Memory::Kind::GPU_DYNAMIC_MEM: break;
  }
  LEGATE_ABORT("Unhandled Processor::Kind ", legate::detail::to_underlying(kind));
}

Processor::Kind to_kind(TaskTarget target)
{
  switch (target) {
    case TaskTarget::GPU: return Processor::Kind::TOC_PROC;
    case TaskTarget::OMP: return Processor::Kind::OMP_PROC;
    case TaskTarget::CPU: return Processor::Kind::LOC_PROC;
  }
  LEGATE_ABORT("Unhandled TaskTarget ", legate::detail::to_underlying(target));
}

Processor::Kind to_kind(VariantCode code)
{
  switch (code) {
    case VariantCode::CPU: return Processor::Kind::LOC_PROC;
    case VariantCode::GPU: return Processor::Kind::TOC_PROC;
    case VariantCode::OMP: return Processor::Kind::OMP_PROC;
  }
  LEGATE_ABORT("Unhandled variant code ", legate::detail::to_underlying(code));
}

Memory::Kind to_kind(StoreTarget target)
{
  switch (target) {
    case StoreTarget::SYSMEM: return Memory::Kind::SYSTEM_MEM;
    case StoreTarget::FBMEM: return Memory::Kind::GPU_FB_MEM;
    case StoreTarget::ZCMEM: return Memory::Kind::Z_COPY_MEM;
    case StoreTarget::SOCKETMEM: return Memory::Kind::SOCKET_MEM;
  }
  LEGATE_ABORT("Unhandled StoreTarget ", legate::detail::to_underlying(target));
}

VariantCode to_variant_code(TaskTarget target)
{
  switch (target) {
    case TaskTarget::GPU: return VariantCode::GPU;
    case TaskTarget::OMP: return VariantCode::OMP;
    case TaskTarget::CPU: return VariantCode::CPU;
  }
  LEGATE_ABORT("Unhandled TaskTarget ", legate::detail::to_underlying(target));
}

VariantCode to_variant_code(Processor::Kind kind) { return to_variant_code(to_target(kind)); }

// ==========================================================================================

std::vector<Legion::DimensionKind> DimOrdering::generate_legion_dims(std::uint32_t ndim) const
{
  std::vector<Legion::DimensionKind> ordering;

  LEGATE_CHECK(ndim > 0);
  ordering.reserve(static_cast<std::size_t>(ndim) + 1);
  switch (kind) {
    case Kind::C: {
      for (auto dim = static_cast<std::int32_t>(ndim) - 1; dim >= 0; --dim) {
        ordering.push_back(static_cast<Legion::DimensionKind>(LEGION_DIM_X + dim));
      }
      break;
    }
    case Kind::FORTRAN: {
      for (std::uint32_t dim = 0; dim < ndim; ++dim) {
        ordering.push_back(static_cast<Legion::DimensionKind>(LEGION_DIM_X + dim));
      }
      break;
    }
    case Kind::CUSTOM: {
      LEGATE_CHECK(ndim == dims.size());
      for (auto dim : dims) {
        ordering.push_back(static_cast<Legion::DimensionKind>(LEGION_DIM_X + dim));
      }
      break;
    }
  }
  // After listing all dimension names, we add the field dimension as the last one so the instance
  // will have an SOA layout
  ordering.push_back(LEGION_DIM_F);

  return ordering;
}

std::vector<Legion::DimensionKind> DimOrdering::generate_legion_dims(const Store& store) const
{
  // If the store is not transformed, we don't have to go through the complex inversion logic below
  if (!store.transformed()) {
    return generate_legion_dims(std::max<>(std::int32_t{1}, store.dim()));
  }

  // We want to enforce user's intent by using the LogicalStore's dimension
  // ordering to determine RegionField's dimension ordering. A compelling example
  // being the desire to reorder the dims of a transposed store such that the new
  // lowest dimension is physically contiguous. To achieve this:
  // 1. We generate initial integer dims using the LogicalStore's dim count
  // 2. Feed them through the store's transform stack via invert_dims(). This is
  //    needed because LogicalStore's dim ordering and count can be different from the
  //    RegionField due to store transforms.
  // 3. Convert the resulting integer dims to Legion dims

  // Generates a tuple of dimension indices in the order in which they should be laid out in the
  // instance. A tuple (0, 1, ..., N - 1) denotes that dimension 0 is the most rapidly changing
  // dimension in the layout; i.e., store[i, j, k, ...] and store[i + 1, j, k, ...] locate
  // contiguously in the instance (called "Fortran order").
  auto dim_ordering = [&] {
    const auto ndim = store.dim();
    legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM> ret;

    ret.reserve(ndim);
    switch (kind) {
      case Kind::C: {
        for (std::int32_t dim = ndim - 1; dim >= 0; --dim) {
          ret.push_back(dim);
        }
        return ret;
      }
      case Kind::FORTRAN: {
        for (std::int32_t dim = 0; dim < ndim; ++dim) {
          ret.push_back(dim);
        }
        return ret;
      }
      case Kind::CUSTOM: {
        for (auto dim : dims) {
          ret.push_back(dim);
        }
        return ret;
      }
    }

    return ret;
  }();

  const auto ndim_rf = store.region_field().dim();

  dim_ordering = store.invert_dims(std::move(dim_ordering));
  if (dim_ordering.empty() && ndim_rf == 1) {
    // Special case where empty store is represented by a 1D region field
    dim_ordering.push_back(0);
  } else {
    // Note that the size of dim_ordering can be smaller than the number of dimensions of the region
    // field if the store has any projections.
    LEGATE_ASSERT(static_cast<std::int32_t>(dim_ordering.size()) <= ndim_rf);
    // All dimension indices should be valid
    LEGATE_ASSERT(std::all_of(dim_ordering.begin(), dim_ordering.end(), [ndim_rf](auto&& dim) {
      return dim >= 0 && dim < ndim_rf;
    }));
  }

  std::vector<Legion::DimensionKind> legion_dims{};

  legion_dims.reserve(dim_ordering.size() + 1);
  std::transform(
    dim_ordering.begin(),
    dim_ordering.end(),
    std::back_inserter(legion_dims),
    [](const std::int32_t d) { return static_cast<Legion::DimensionKind>(d + LEGION_DIM_X); });

  // After listing all dimension names, we add the field dimension as the last one so the instance
  // will have an SOA layout
  legion_dims.push_back(LEGION_DIM_F);

  return legion_dims;
}

// ==========================================================================================

StoreMapping::StoreMapping(InstanceMappingPolicy policy, const Store* store)
  : StoreMapping{std::move(policy), {&store, 1}}
{
}

StoreMapping::StoreMapping(InstanceMappingPolicy policy, Span<const Store* const> stores)
  : policy_{std::move(policy)}, stores_{stores.begin(), stores.end()}
{
}

StoreMapping::StoreMapping(InstanceMappingPolicy policy,
                           Span<const InternalSharedPtr<Store>> stores)
  : policy_{std::move(policy)}
{
  stores_.reserve(stores.size());
  std::transform(stores.begin(), stores.end(), std::back_inserter(stores_), [](const auto& store) {
    return store.get();
  });
}

void StoreMapping::add_store(const Store* store) { stores_.emplace_back(store); }

bool StoreMapping::for_future() const
{
  return std::any_of(
    stores().begin(), stores().end(), [](const Store* store) { return store->is_future(); });
}

bool StoreMapping::for_unbound_store() const
{
  return std::any_of(
    stores().begin(), stores().end(), [](const Store* store) { return store->unbound(); });
}

const Store* StoreMapping::store() const { return stores().front(); }

std::uint32_t StoreMapping::requirement_index() const
{
  if (stores().empty()) {
    constexpr std::uint32_t INVALID = -1U;

    return INVALID;
  }

  const auto idx = store()->requirement_index();

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    const auto all_req_idx_equal =
      std::all_of(stores().begin(), stores().end(), [&](const auto* store) {
        return store->requirement_index() == idx;
      });

    LEGATE_CHECK(all_req_idx_equal);
  }
  return idx;
}

std::set<std::uint32_t> StoreMapping::requirement_indices() const
{
  std::set<std::uint32_t> indices;

  for (auto&& store : stores()) {
    if (!store->is_future()) {
      indices.insert(store->region_field().index());
    }
  }
  return indices;
}

std::set<const Legion::RegionRequirement*> StoreMapping::requirements() const
{
  std::set<const Legion::RegionRequirement*> reqs;

  for (auto&& store : stores()) {
    if (store->is_future()) {
      continue;
    }

    if (const auto& req = store->region_field().get_requirement(); req.region.exists()) {
      reqs.insert(&req);
    }
  }
  return reqs;
}

void StoreMapping::populate_layout_constraints(
  Legion::LayoutConstraintSet& layout_constraints) const
{
  if (stores().size() > 1) {
    // We assume that all stores in the mapping have the same number of dimensions
    // at least, and that the other stores follow the dimension ordering of the
    // first store, which may be adjusted due to transforms, such as, transpose.
    LEGATE_ASSERT(std::all_of(stores().begin(), stores().end(), [&](const auto* st) {
      return st->dim() == store()->dim();
    }));
  }

  // This 2-step is necessary because Legion::OrderingConstraint constructor takes the vector
  // by const-ref. Similarly, layout_constraints.add_constraint() ALSO takes its
  // OrderingConstraint by const-ref, meaning that if we go the usual route of
  //
  // layout_constraints.add_constraint(Legion::OrderingConstraint{dimension_ordering, false})
  //
  // ...would result in not 1, not 2, but 3 deep copies of our vector! Now it's a 0-copy
  // operation (a move)
  if (policy().ordering.has_value()) {
    const auto& ordering = policy().ordering->impl();

    layout_constraints.ordering_constraint.ordering = ordering->generate_legion_dims(*store());
  }
  layout_constraints.ordering_constraint.contiguous = false;
  layout_constraints.add_constraint(Legion::MemoryConstraint{to_kind(policy().target)});

  if ((stores().size() > 1) && LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    std::unordered_set<Legion::FieldID> field_set{};

    for (auto&& st : stores()) {
      field_set.insert(st->region_field().field_id());
    }
    // Due to removing InstLayout::AOS (and therefore defaulting to Instlayout::SOA), we should
    // only ever have 1 field (probably). See
    // https://github.com/nv-legate/legate.internal/pull/2514#pullrequestreview-3026966414.
    LEGATE_CHECK(field_set.size() == 1);
  }

  // To silence gcc 12 error:
  //
  // c++/bits/stl_algobase.h:434:30: error: 'void* __builtin_memcpy(void*, const void*, long
  // unsigned int)' forming offset 4 is out of the bounds [0, 4] of object '<anonymous>' with
  // type 'const unsigned int [1]' [-Werror=array-bounds]
  // when implementing it like:
  // layout_constraints.field_constraint.field_set  = {first_region_field.field_id()};
  layout_constraints.field_constraint.field_set.clear();
  layout_constraints.field_constraint.field_set.push_back(store()->region_field().field_id());
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
  return std::make_unique<detail::StoreMapping>(std::move(policy), store);
}

}  // namespace legate::mapping::detail
