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

#include "legate/mapping/detail/base_mapper.h"

#include "legate/mapping/detail/instance_manager.h"
#include "legate/mapping/detail/mapping.h"
#include "legate/mapping/detail/operation.h"
#include "legate/mapping/detail/store.h"
#include "legate/mapping/operation.h"
#include "legate/runtime/detail/projection.h"
#include "legate/runtime/detail/runtime.h"
#include "legate/runtime/detail/shard.h"
#include "legate/task/detail/task_return.h"
#include "legate/utilities/detail/core_ids.h"
#include "legate/utilities/detail/enumerate.h"
#include "legate/utilities/detail/env.h"
#include "legate/utilities/detail/formatters.h"
#include "legate/utilities/detail/store_iterator_cache.h"
#include "legate/utilities/detail/type_traits.h"
#include "legate/utilities/detail/zip.h"
#include <legate/utilities/detail/linearize.h>

#include "mappers/mapping_utilities.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <functional>
#include <limits>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace legate::mapping::detail {

namespace {

const std::vector<StoreTarget>& default_store_targets(Processor::Kind kind,
                                                      bool for_pool_size = false)
{
  switch (kind) {
    case Processor::Kind::TOC_PROC: {
      static const std::vector<StoreTarget> targets = {StoreTarget::FBMEM, StoreTarget::ZCMEM};
      return targets;
    }
    case Processor::Kind::OMP_PROC: {
      // TODO(wonchanl): the distinction should go away once Realm drops the socket memory as a
      // separate entity
      static const std::vector<StoreTarget> targets = {StoreTarget::SOCKETMEM, StoreTarget::SYSMEM};
      static const std::vector<StoreTarget> target_for_pool_size = {StoreTarget::SOCKETMEM};
      return for_pool_size ? target_for_pool_size : targets;
    }
    case Processor::Kind::LOC_PROC: {
      static const std::vector<StoreTarget> targets = {StoreTarget::SYSMEM};
      return targets;
    }
    default: break;
  }
  LEGATE_ABORT(
    "Could not find ProcessorKind ", static_cast<int>(kind), " in default store targets");
}

std::string log_mappable(const Legion::Mappable& mappable, bool prefix_only = false)
{
  static const std::unordered_map<Legion::MappableType, std::string> prefixes = {
    {LEGION_TASK_MAPPABLE, "Task "},
    {LEGION_COPY_MAPPABLE, "Copy "},
    {LEGION_INLINE_MAPPABLE, "Inline mapping "},
    {LEGION_PARTITION_MAPPABLE, "Partition "},
  };
  auto finder = prefixes.find(mappable.get_mappable_type());
  LEGATE_ASSERT(finder != prefixes.end());
  if (prefix_only) {
    return finder->second;
  }

  return fmt::format("{}{}", finder->second, mappable.get_unique_id());
}

}  // namespace

Legion::VariantID BaseMapper::select_task_variant_(Legion::Mapping::MapperContext ctx,
                                                   const Legion::Task& task,
                                                   const Processor& proc)
{
  const auto variant = find_variant_(ctx, task, proc.kind());

  LEGATE_CHECK(variant.has_value());
  return *variant;
}

// ==========================================================================================

BaseMapper::BaseMapper()
  : Mapper{Legion::Runtime::get_runtime()->get_mapper_runtime()},
    mapper_name_{
      fmt::format("{} on Node {}",
                  legate::detail::Runtime::get_runtime()->core_library()->get_library_name(),
                  local_machine_.node_id)}
{
}

BaseMapper::~BaseMapper()
{
  if (legate::detail::LEGATE_SHOW_USAGE.get().value_or(false)) {
    constexpr std::string_view memory_kinds[] = {
#define MEM_NAMES(name, desc) LEGATE_STRINGIZE(name),
      REALM_MEMORY_KINDS(MEM_NAMES)
#undef MEM_NAMES
    };
    const auto mapper_name = std::string_view{get_mapper_name()};

    for (auto&& [mem, num_bytes] : local_instances_.aggregate_instance_sizes()) {
      const auto capacity = mem.capacity();
      const auto percent  = 100.0 * static_cast<double>(num_bytes) / static_cast<double>(capacity);

      logger().print() << fmt::format(
        "{} used {} bytes of {} memory {:#04x} with {} total bytes ({:.2g}%)",
        mapper_name,
        num_bytes,
        memory_kinds[mem.kind()],
        mem.id,
        capacity,
        percent);
    }
  }
}

namespace {

void populate_input_collective_regions(
  const Legion::Task& legion_task,
  const Task& legate_task,
  const legate::detail::StoreIteratorCache<InternalSharedPtr<Store>>& get_stores,
  std::set<unsigned>* check_collective_regions)
{
  const auto hi          = legion_task.index_domain.hi();
  const auto lo          = legion_task.index_domain.lo();
  const auto task_volume = legion_task.index_domain.get_volume();

  for (auto&& array : legate_task.inputs()) {
    for (auto&& store : get_stores(*array)) {
      if (store->is_future()) {
        continue;
      }

      const auto idx = store->requirement_index();

      if ((task_volume > 1) &&
          legion_task.regions[idx].partition == Legion::LogicalPartition::NO_PART) {
        check_collective_regions->insert(idx);
        continue;
      }

      for (auto&& d : store->find_imaginary_dims()) {
        if ((hi[d] - lo[d]) >= 1) {
          check_collective_regions->insert(idx);
          break;
        }
      }
    }
  }
}

void populate_reduction_collective_regions(
  const Legion::Task& legion_task,
  const Task& legate_task,
  const legate::detail::StoreIteratorCache<InternalSharedPtr<Store>>& get_stores,
  std::set<unsigned>* check_collective_regions)
{
  for (auto&& array : legate_task.reductions()) {
    for (auto&& store : get_stores(*array)) {
      if (store->is_future()) {
        continue;
      }

      const auto idx = store->requirement_index();
      auto&& req     = legion_task.regions[idx];

      if (req.privilege & LEGION_WRITE_PRIV) {
        continue;
      }
      if (req.handle_type == LEGION_SINGULAR_PROJECTION || req.projection != 0) {
        check_collective_regions->insert(idx);
      }
    }
  }
}

}  // namespace

void BaseMapper::select_task_options(Legion::Mapping::MapperContext ctx,
                                     const Legion::Task& task,
                                     TaskOptions& output)
{
  Task legate_task{&task, runtime, ctx};

  {
    auto get_stores = legate::detail::StoreIteratorCache<InternalSharedPtr<Store>>{};

    populate_input_collective_regions(
      task, legate_task, get_stores, &output.check_collective_regions);
    populate_reduction_collective_regions(
      task, legate_task, get_stores, &output.check_collective_regions);
  }

  const auto target = [&] {
    const auto& machine_desc = legate_task.machine();
    const auto& targets      = machine_desc.valid_targets();
    const auto it            = std::find_if(
      targets.begin(), targets.end(), [&](TaskTarget tgt) { return has_variant_(ctx, task, tgt); });

    if (it == targets.end()) {
      LEGATE_ABORT("Task ",
                   task.get_task_name(),
                   "[",
                   task.get_provenance_string(),
                   "] does not have a valid variant for this resource configuration: ",
                   machine_desc);
    }
    return *it;
  }();

  // The initial processor just needs to have the same kind as the eventual target of this task
  output.initial_proc = local_machine_.procs(target).front();

  // We never want valid instances
  output.valid_instances = false;
}

void BaseMapper::premap_task(Legion::Mapping::MapperContext /*ctx*/,
                             const Legion::Task& /*task*/,
                             const PremapTaskInput& /*input*/,
                             PremapTaskOutput& /*output*/)
{
  // NO-op since we know that all our futures should be mapped in the system memory
}

void BaseMapper::slice_task(Legion::Mapping::MapperContext ctx,
                            const Legion::Task& task,
                            const SliceTaskInput& input,
                            SliceTaskOutput& output)
{
  const Task legate_task{&task, runtime, ctx};

  auto&& machine_desc = legate_task.machine();
  auto local_range    = local_machine_.slice(legate_task.target(), machine_desc);

  Legion::ProjectionID projection = 0;
  for (auto&& req : task.regions) {
    if (req.tag == traits::detail::to_underlying(legate::detail::CoreMappingTag::KEY_STORE)) {
      projection = req.projection;
      break;
    }
  }
  auto* key_functor = legate::detail::find_projection_function(projection);

  // For multi-node cases we should already have been sharded so we
  // should just have one or a few points here on this node, so iterate
  // them and round-robin them across the local processors here
  output.slices.reserve(input.domain.get_volume());

  // Get the domain for the sharding space also
  Domain sharding_domain = task.index_domain;
  if (task.sharding_space.exists()) {
    sharding_domain = runtime->get_index_space_domain(ctx, task.sharding_space);
  }

  const auto lo                = key_functor->project_point(sharding_domain.lo());
  const auto hi                = key_functor->project_point(sharding_domain.hi());
  const auto start_proc_id     = machine_desc.processor_range().low;
  const auto total_tasks_count = legate::detail::linearize(lo, hi, hi) + 1;

  for (Domain::DomainPointIterator itr{input.domain}; itr; ++itr) {
    const auto p = key_functor->project_point(itr.p);
    const auto idx =
      (legate::detail::linearize(lo, hi, p) * local_range.total_proc_count() / total_tasks_count) +
      start_proc_id;

    output.slices.emplace_back(Domain{itr.p, itr.p},
                               local_range[static_cast<std::uint32_t>(idx)],
                               false /*recurse*/,
                               false /*stealable*/);
  }
}

bool BaseMapper::has_variant_(Legion::Mapping::MapperContext ctx,
                              const Legion::Task& task,
                              TaskTarget target)
{
  return find_variant_(ctx, task, to_kind(target)).has_value();
}

std::optional<Legion::VariantID> BaseMapper::find_variant_(Legion::Mapping::MapperContext ctx,
                                                           const Legion::Task& task,
                                                           Processor::Kind kind)
{
  const auto key            = VariantCacheKey{task.task_id, kind};
  const auto old_cache_size = variants_.size();

  if (const auto it = variants_.find(key); it != variants_.end()) {
    return it->second;
  }

  std::vector<Legion::VariantID> avail_variants;

  runtime->find_valid_variants(ctx, key.first, avail_variants, key.second);
  if (variants_.size() != old_cache_size) {
    // If the size has changed, that means that while we made the runtime call another mapper
    // call pre-empted us and ran this function. It's not guaranteed, but there is a
    // possibility that the call filled out the exact value we are looking for, so check the
    // cache one more time...
    if (const auto it = variants_.find(key); it != variants_.end()) {
      return it->second;
    }
  }
  // ...otherwise a true miss. Insert our new value into the cache
  std::optional<Legion::VariantID> ret = std::nullopt;

  for (auto vid : avail_variants) {
    LEGATE_ASSERT(vid > 0);
    switch (VariantCode{vid}) {
      case VariantCode::CPU: [[fallthrough]];
      case VariantCode::OMP: [[fallthrough]];
      case VariantCode::GPU: ret = vid; break;
      default: LEGATE_ABORT("Unhandled variant kind ", vid);  // unhandled variant kind
    }
  }
  return variants_.emplace(key, std::move(ret)).first->second;
}

namespace {

void validate_colocation(const std::vector<mapping::StoreMapping>& client_mappings,
                         const char* mapper_name)
{
  for (auto&& client_mapping : client_mappings) {
    const auto* mapping = client_mapping.impl();
    auto&& stores       = mapping->stores;

    LEGATE_CHECK(!stores.empty());
    LEGATE_CHECK(!(mapping->for_future() || mapping->for_unbound_store()) || stores.size() == 1);

    const auto cant_colocate = [&first_store = *stores.front()](const Store* store) {
      return !store->can_colocate_with(first_store);
    };
    if (std::any_of(stores.cbegin() + 1, stores.cend(), cant_colocate)) {
      LEGATE_ABORT("Mapper ", mapper_name, " tried to colocate stores that cannot colocate");
    }
  }
}  // namespace

struct MappingData {
  std::unordered_map<std::uint32_t, const StoreMapping*> mapped_futures{};
  std::vector<std::unique_ptr<StoreMapping>> for_futures{};
  std::unordered_set<RegionField::Id, hasher<RegionField::Id>> mapped_regions{};
  std::vector<std::unique_ptr<StoreMapping>> for_unbound_stores{};
  std::vector<std::unique_ptr<StoreMapping>> for_stores{};
};

[[nodiscard]] MappingData handle_client_mappings(
  mapping::StoreMapping::ReleaseKey key,
  const char* mapper_name,
  std::vector<mapping::StoreMapping>&& client_mappings,
  const Legion::Task& task)
{
  // Move into local to "complete" the move
  auto cmappings = std::move(client_mappings);
  MappingData ret{};

  for (auto&& client_mapping : cmappings) {
    const auto* mapping = client_mapping.impl();

    if (mapping->for_future()) {
      const auto fut_idx = mapping->store()->future_index();
      // Only need to map Future-backed Stores corresponding to inputs (i.e. one of task.futures)
      if (fut_idx >= task.futures.size()) {
        continue;
      }

      const auto [it, inserted] = ret.mapped_futures.try_emplace(fut_idx, mapping);

      if (inserted) {
        ret.for_futures.emplace_back(client_mapping.release_(key));
      } else if (it->second->policy != mapping->policy) {
        LEGATE_ABORT("Mapper ", mapper_name, " returned duplicate store mappings");
      }
    } else if (mapping->for_unbound_store()) {
      ret.mapped_regions.insert(mapping->store()->unique_region_field_id());
      ret.for_unbound_stores.emplace_back(client_mapping.release_(key));
    } else {
      for (const auto* store : mapping->stores) {
        ret.mapped_regions.insert(store->unique_region_field_id());
      }
      ret.for_stores.emplace_back(client_mapping.release_(key));
    }
  }
  return ret;
}

void validate_policies(const std::vector<std::unique_ptr<StoreMapping>>& for_stores,
                       const char* mapper_name)
{
  std::unordered_map<RegionField::Id, InstanceMappingPolicy, hasher<RegionField::Id>> policies;

  for (const auto& mapping : for_stores) {
    const auto& policy = mapping->policy;

    for (const auto& store : mapping->stores) {
      const auto key            = store->unique_region_field_id();
      const auto [it, inserted] = policies.try_emplace(key, policy);

      if (!inserted && (policy != it->second)) {
        LEGATE_ABORT("Mapper ", mapper_name, " returned inconsistent store mappings");
      }
    }
  }
}

[[nodiscard]] MappingData initialize_mapping_categories(
  mapping::StoreMapping::ReleaseKey key,
  std::vector<mapping::StoreMapping>&& client_mappings,
  const char* mapper_name,
  const Legion::Task& task)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    validate_colocation(client_mappings, mapper_name);
  }

  auto ret = handle_client_mappings(key, mapper_name, std::move(client_mappings), task);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    validate_policies(ret.for_stores, mapper_name);
  }

  return ret;
}

void generate_default_mappings(
  const std::vector<InternalSharedPtr<Array>>& arrays,
  StoreTarget default_option,
  const Legion::Task& legion_task,
  std::unordered_map<std::uint32_t, const StoreMapping*>* mapped_futures,
  std::vector<std::unique_ptr<StoreMapping>>* for_futures,
  std::unordered_set<RegionField::Id, hasher<RegionField::Id>>* mapped_regions,
  std::vector<std::unique_ptr<StoreMapping>>* for_unbound_stores,
  std::vector<std::unique_ptr<StoreMapping>>* for_stores)
{
  const auto get_stores = legate::detail::StoreIteratorCache<InternalSharedPtr<Store>>{};

  for (auto&& array : arrays) {
    for (auto&& store : get_stores(*array)) {
      if (store->is_future()) {
        const auto fut_idx = store->future_index();
        // Only need to map Future-backed Stores corresponding to inputs (i.e. one of
        // task.futures)
        if (fut_idx >= legion_task.futures.size()) {
          continue;
        }

        if (const auto [it, inserted] = mapped_futures->try_emplace(fut_idx); inserted) {
          auto mapping = StoreMapping::default_mapping(store.get(), default_option);

          it->second = for_futures->emplace_back(std::move(mapping)).get();
        }
      } else {
        const auto [_, inserted] = mapped_regions->emplace(store->unique_region_field_id());

        if (inserted) {
          auto mapping = StoreMapping::default_mapping(store.get(), default_option);

          if (store->unbound()) {
            for_unbound_stores->push_back(std::move(mapping));
          } else {
            for_stores->push_back(std::move(mapping));
          }
        }
      }
    }
  }
}

void map_future_backed_stores(
  const std::unordered_map<std::uint32_t, const StoreMapping*>& mapped_futures,
  const std::vector<std::unique_ptr<StoreMapping>>& for_futures,
  const LocalMachine& local_machine,
  const Legion::Task& task,
  Legion::Mapping::Mapper::MapTaskOutput* output)
{
  LEGATE_CHECK(mapped_futures.size() <= task.futures.size());
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    // The launching code should be packing all Store-backing Futures first.
    for (auto&& [idx, mapping] : legate::detail::enumerate(for_futures)) {
      LEGATE_CHECK(mapping->store()->future_index() == idx);
    }
  }

  output->future_locations.resize(mapped_futures.size());
  for (auto&& mapping : for_futures) {
    const auto fut_idx = mapping->store()->future_index();
    StoreTarget target = mapping->policy.target;

    if (LEGATE_DEFINED(LEGATE_NO_FUTURES_ON_FB) && (target == StoreTarget::FBMEM)) {
      target = StoreTarget::ZCMEM;
    }
    output->future_locations[fut_idx] = local_machine.get_memory(task.target_proc, target);
  }
}

void map_unbound_stores(const std::vector<std::unique_ptr<StoreMapping>>& mappings,
                        const LocalMachine& local_machine,
                        const Legion::Processor& target_proc,
                        Legion::Mapping::Mapper::MapTaskOutput* output)
{
  for (auto&& mapping : mappings) {
    const auto* store  = mapping->store();
    const auto req_idx = mapping->requirement_index();

    output->output_targets[req_idx] = local_machine.get_memory(target_proc, mapping->policy.target);

    const auto ndim = mapping->store()->dim();
    std::vector<Legion::DimensionKind> dimension_ordering;

    dimension_ordering.reserve(static_cast<std::size_t>(ndim) + 1);
    for (auto dim = ndim - 1; dim >= 0; --dim) {
      dimension_ordering.push_back(static_cast<Legion::DimensionKind>(
        static_cast<std::int32_t>(Legion::DimensionKind::LEGION_DIM_X) + dim));
    }
    dimension_ordering.push_back(Legion::DimensionKind::LEGION_DIM_F);

    auto& output_constraint   = output->output_constraints[req_idx];
    auto& ordering_constraint = output_constraint.ordering_constraint;

    ordering_constraint.ordering   = std::move(dimension_ordering);
    ordering_constraint.contiguous = false;

    output_constraint.alignment_constraints.emplace_back(
      store->region_field().field_id(), LEGION_EQ_EK, store->type()->alignment());
  }
}

[[nodiscard]] std::size_t calculate_legate_allocation_size(Legion::Mapping::MapperRuntime* runtime,
                                                           Legion::Mapping::MapperContext ctx,
                                                           const Legion::Task& task,
                                                           const Task& legate_task,
                                                           const VariantInfo& vinfo)
{
  auto&& all_arrays = {std::cref(legate_task.inputs()),
                       std::cref(legate_task.outputs()),
                       std::cref(legate_task.reductions())};
  auto get_stores   = legate::detail::StoreIteratorCache<InternalSharedPtr<Store>>{};

  // Here we calculate the total size of buffers created for scalar stores and unbound stores in the
  // following way: each output or reduction scalar store embeds a buffer to hold the update; each
  // unbound store gets a buffer that holds the number of elements, which will later be retrieved to
  // keep track of the store's key partition (of the legate::detail::Weighted type).
  auto layout = legate::detail::TaskReturnLayoutForUnpack{};
  for (auto&& arrays : all_arrays) {
    for (auto&& array : arrays.get()) {
      for (auto&& store : get_stores(*array)) {
        if (store->is_future()) {
          const auto& type = store->type();
          if (type->size() == 0) {
            continue;
          }
          std::ignore = layout.next(type->size(), type->alignment());
        } else if (store->unbound()) {
          // The number of elements for each unbound store is stored in a buffer of type
          // std::size_t
          std::ignore = layout.next(sizeof(std::size_t), alignof(std::size_t));
        }
      }
    }
  }
  // The math above only accounts for the buffers that are created when the stores are deserialized
  // in the task preamble. Once the task is done, the final values in those buffers are packed into
  // a return buffer of the task, which is yet to be included in the pool size. Therefore, we double
  // the pool size calculated so far to take the return buffer size into account.
  const auto size = layout.total_size() * 2;

  // If the aggregate size of buffers is greater than the return size of the variant, raise an
  // error
  //
  // TODO(wonchanl): the task can still exceed the return size if it has an exception string
  // longer than the specified return size minus the space reserved for scalar returns. Currently,
  // Legate defers the error handling to Legion in this case, but ideally Legate should handle it.
  if (size > vinfo.options.return_size) {
    LEGATE_ABORT(fmt::format(
      "Necessary pool size ({} bytes) exceeds the specified return size ({} bytes) of task {}[{}]",
      size,
      vinfo.options.return_size,
      Legion::Mapping::Utilities::to_string(runtime, ctx, task),
      task.get_provenance_string()));
  }

  // If the task can raise an exception, we reserve space for the serialized exception by
  // requesting a pool as big as the return size
  //
  // TODO(wonchanl): this return size business is brittle and also inscrutable for users because
  // they don't necessarily know what constitutes the buffer returned from a variant. Legate
  // should instead infer the right return size based on a task signature.
  return legate_task.can_raise_exception() ? vinfo.options.return_size : size;
}

void calculate_pool_sizes(Legion::Mapping::MapperRuntime* runtime,
                          Legion::Mapping::MapperContext ctx,
                          const Legion::Task& task,
                          const LocalMachine& local_machine,
                          Logger* logger,
                          Task* legate_task,
                          Legion::Mapping::Mapper::MapTaskOutput* output)
{
  auto* library = legate_task->library();
  // TODO(mpapadakis): Unify the variant finding with BaseMapper::find_variant_
  auto&& maybe_vinfo = library->find_task(legate_task->task_id())
                         ->find_variant(to_variant_code(legate_task->target()));

  LEGATE_CHECK(maybe_vinfo.has_value());

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  auto&& vinfo = maybe_vinfo->get();

  const auto legate_alloc_size =
    calculate_legate_allocation_size(runtime, ctx, task, *legate_task, vinfo);

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    logger->debug() << "Task " << Legion::Mapping::Utilities::to_string(runtime, ctx, task)
                    << " needs at least " << legate_alloc_size
                    << " bytes in a pool for scalar stores, unbound stores, and exceptions";
  }

  const auto& allocation_pool_targets = default_store_targets(task.target_proc.kind(), true);
  if (!vinfo.options.has_allocations) {
    // Unfortunately, even when the user said her task doesn't create any allocations, there can be
    // some made by Legate. Therefore, we still need to return the right bounds so Legate can create
    // those allocations in (the pre- and post-amble of) the user task.
    for (auto&& target : allocation_pool_targets) {
      output->leaf_pool_bounds.try_emplace(local_machine.get_memory(task.target_proc, target),
                                           legate_alloc_size);
    }
    return;
  }

  auto* legate_mapper = library->get_mapper();
  for (auto&& target : allocation_pool_targets) {
    const auto size   = legate_mapper->allocation_pool_size(mapping::Task{legate_task}, target);
    const auto memory = local_machine.get_memory(task.target_proc, target);
    output->leaf_pool_bounds.try_emplace(
      memory,
      size.has_value()
        ? Legion::PoolBounds{*size + legate_alloc_size}
        : Legion::PoolBounds{Legion::UnboundPoolScope::LEGION_INDEX_TASK_UNBOUNDED_POOL,
                             std::numeric_limits<std::size_t>::max()});
  }
}

}  // namespace

void BaseMapper::map_task(Legion::Mapping::MapperContext ctx,
                          const Legion::Task& task,
                          const MapTaskInput& /*input*/,
                          MapTaskOutput& output)
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    logger().debug() << "Entering map_task for "
                     << Legion::Mapping::Utilities::to_string(runtime, ctx, task);
  }

  // Should never be mapping the top-level task here
  LEGATE_CHECK(task.get_depth() > 0);

  // Let's populate easy outputs first
  output.chosen_variant = select_task_variant_(ctx, task, task.target_proc);

  Task legate_task{&task, runtime, ctx};

  output.task_priority      = legate_task.priority();
  output.copy_fill_priority = legate_task.priority();

  if (task.is_index_space) {
    // If this is an index task, point tasks already have the right targets, so we just need to
    // copy them to the mapper output
    output.target_procs.push_back(task.target_proc);
  } else {
    // If this is a single task, here is the right place to compute the final target processor
    const auto local_range =
      local_machine_.slice(legate_task.target(), legate_task.machine(), task.local_function);

    LEGATE_ASSERT(!local_range.empty());
    output.target_procs.push_back(local_range.first());
  }

  const auto& options = default_store_targets(task.target_proc.kind());

  auto [mapped_futures, for_futures, mapped_regions, for_unbound_stores, for_stores] = [&] {
    auto client_mappings =
      legate_task.library()->get_mapper()->store_mappings(mapping::Task{&legate_task}, options);

    return initialize_mapping_categories(
      mapping::StoreMapping::ReleaseKey{}, std::move(client_mappings), get_mapper_name(), task);
  }();

  // Generate default mappings for stores that are not yet mapped by the client mapper
  const auto default_option = options.front();
  for (auto&& arr : {std::cref(legate_task.inputs()),
                     std::cref(legate_task.outputs()),
                     std::cref(legate_task.reductions())}) {
    generate_default_mappings(arr,
                              default_option,
                              task,
                              &mapped_futures,
                              &for_futures,
                              &mapped_regions,
                              &for_unbound_stores,
                              &for_stores);
  }

  // Map future-backed stores
  map_future_backed_stores(mapped_futures, for_futures, local_machine_, task, &output);
  // Map unbound stores
  map_unbound_stores(for_unbound_stores, local_machine_, task.target_proc, &output);

  OutputMap output_map;

  output_map.reserve(task.regions.size());
  output.chosen_instances.resize(task.regions.size());
  for (auto&& [task_region, chosen_instance] :
       legate::detail::zip_equal(task.regions, output.chosen_instances)) {
    output_map[&task_region] = &chosen_instance;
  }

  map_legate_stores_(ctx,
                     task,
                     for_stores,
                     task.target_proc,
                     output_map,
                     legate_task.machine().count() < task.index_domain.get_volume());

  calculate_pool_sizes(runtime, ctx, task, local_machine_, &logger(), &legate_task, &output);

  for (auto&& mapping : for_stores) {
    if (!mapping->policy.redundant) {
      continue;
    }

    for (auto req_idx : mapping->requirement_indices()) {
      if (task.regions[req_idx].privilege != LEGION_READ_ONLY) {
        continue;
      }
      output.untracked_valid_regions.insert(req_idx);
    }
  }
}

void BaseMapper::replicate_task(Legion::Mapping::MapperContext /*ctx*/,
                                const Legion::Task& /*task*/,
                                const ReplicateTaskInput& /*input*/,
                                ReplicateTaskOutput& /*output*/)

{
  LEGATE_ABORT("Should not be called");
}

void BaseMapper::map_legate_stores_(Legion::Mapping::MapperContext ctx,
                                    const Legion::Mappable& mappable,
                                    std::vector<std::unique_ptr<StoreMapping>>& mappings,
                                    Processor target_proc,
                                    OutputMap& output_map,
                                    bool overdecomposed)
{
  auto try_mapping = [&](bool can_fail) {
    const Legion::Mapping::PhysicalInstance NO_INST{};
    std::vector<Legion::Mapping::PhysicalInstance> instances;

    instances.reserve(mappings.size());
    for (auto&& mapping : mappings) {
      Legion::Mapping::PhysicalInstance result = NO_INST;
      auto reqs                                = mapping->requirements();
      // Point tasks collectively writing to the same region must be doing so via distinct
      // instances. This contract is somewhat difficult to satisfy while the mapper also tries to
      // reuse the existing instance for the region, because when the tasks are mapped to processors
      // with a shared memory, the mapper should reuse the instance for only one of the tasks and
      // not for the others, a logic that is tedious to write correctly. For that reason, we simply
      // give up on reusing instances for regions used in collective writes whenever more than one
      // task can try to reuse the existing instance for the same region. The obvious case where the
      // mapepr can safely reuse the instances is that the region is mapped to a framebuffer and not
      // over-decomposed (i.e., there's a 1-1 mapping between tasks and GPUs).
      const auto must_alloc_collective_writes =
        mappable.get_mappable_type() == Legion::Mappable::TASK_MAPPABLE &&
        (overdecomposed || mapping->policy.target != StoreTarget::FBMEM);
      while (map_legate_store_(ctx,
                               mappable,
                               *mapping,
                               reqs,
                               target_proc,
                               result,
                               can_fail,
                               must_alloc_collective_writes)) {
        if (NO_INST == result) {
          LEGATE_ASSERT(can_fail);
          for (auto&& instance : instances) {
            runtime->release_instance(ctx, instance);
          }
          return false;
        }
        std::stringstream reqs_ss;
        if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
          for (auto req_idx : mapping->requirement_indices()) {
            reqs_ss << " " << req_idx;
          }
        }
        if (runtime->acquire_instance(ctx, result)) {
          if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
            logger().debug() << log_mappable(mappable) << ": acquired instance " << result
                             << " for reqs:" << reqs_ss.str();
          }
          break;
        }
        if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
          logger().debug() << log_mappable(mappable) << ": failed to acquire instance " << result
                           << " for reqs:" << reqs_ss.str();
        }

        if ((*reqs.begin())->redop == 0) {
          static_cast<void>(local_instances_.erase(result));
        } else {
          static_cast<void>(reduction_instances_.erase(result));
        }

        // We have deleted this instance from our caches, no need to stay subscribed
        runtime->unsubscribe(ctx, result);
        creating_operation_.erase(result);
        result = NO_INST;
      }
      instances.push_back(result);
    }

    // If we're here, all stores are mapped and instances are all acquired
    for (std::uint32_t idx = 0; idx < mappings.size(); ++idx) {
      auto& mapping  = mappings[idx];
      auto& instance = instances[idx];
      for (auto&& req : mapping->requirements()) {
        output_map[req]->push_back(instance);
      }
    }
    return true;
  };

  // We can retry the mapping with tightened policies only if at least one of the policies
  // is lenient
  bool can_fail = false;
  for (auto&& mapping : mappings) {
    can_fail = can_fail || !mapping->policy.exact;
  }

  if (!try_mapping(can_fail)) {
    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      logger().debug() << log_mappable(mappable) << " failed to map all stores, retrying with "
                       << "tighter policies";
    }
    // If instance creation failed we try mapping all stores again, but request tight instances for
    // write requirements. The hope is that these write requirements cover the entire region (i.e.
    // they use a complete partition), so the new tight instances will invalidate any pre-existing
    // "bloated" instances for the same region, freeing up enough memory so that mapping can succeed
    tighten_write_policies_(mappable, mappings);
    try_mapping(false);
  }
}

void BaseMapper::tighten_write_policies_(const Legion::Mappable& mappable,
                                         const std::vector<std::unique_ptr<StoreMapping>>& mappings)
{
  for (auto&& mapping : mappings) {
    // If the policy is exact, there's nothing we can tighten
    if (mapping->policy.exact) {
      continue;
    }

    auto priv = traits::detail::to_underlying(LEGION_NO_ACCESS);
    for (const auto* req : mapping->requirements()) {
      priv |= req->privilege;
    }
    // We tighten only write requirements
    if (!(priv & LEGION_WRITE_PRIV)) {
      continue;
    }

    if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
      std::stringstream reqs_ss;
      for (auto req_idx : mapping->requirement_indices()) {
        reqs_ss << " " << req_idx;
      }
      logger().debug() << log_mappable(mappable)
                       << ": tightened mapping policy for reqs:" << std::move(reqs_ss).str();
    }
    mapping->policy.exact = true;
  }
}

bool BaseMapper::map_reduction_instance_(const Legion::Mapping::MapperContext& ctx,
                                         const Legion::Mappable& mappable,
                                         const Processor& target_proc,
                                         const std::vector<Legion::FieldID>& fields,
                                         const std::vector<Legion::LogicalRegion>& regions,
                                         const InstanceMappingPolicy& policy,
                                         Memory target_memory,
                                         GlobalRedopID redop,
                                         Legion::LayoutConstraintSet* layout_constraints,
                                         Legion::Mapping::PhysicalInstance* result,
                                         bool* need_acquire,
                                         std::size_t* footprint)
{
  *footprint    = 0;
  *need_acquire = true;

  const auto can_cache = target_proc.kind() == Processor::TOC_PROC && fields.size() == 1 &&
                         regions.size() == 1 && policy.allocation != AllocPolicy::MUST_ALLOC;

  // reuse reductions only for GPU tasks:
  if (can_cache) {
    // See if we already have it in our local instances
    auto ret = reduction_instances_.find_instance(
      redop, regions.front(), fields.front(), target_memory, policy);

    if (ret.has_value()) {
      *result = *std::move(ret);
      // Needs acquire to keep the runtime happy
      *need_acquire = true;

      if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
        logger().debug() << "Operation " << mappable.get_unique_id()
                         << ": reused cached reduction instance " << result << " for "
                         << regions.front();
      }

      return true /* success */;
    }
  }

  // if we didn't find it, create one
  layout_constraints->add_constraint(Legion::SpecializedConstraint{
    LEGION_AFFINE_REDUCTION_SPECIALIZE, static_cast<Legion::ReductionOpID>(redop)});

  const auto find_or_create_instance = [&] {
    if (!can_cache) {
      // The instance will be acquired during creation
      *need_acquire = false;
      return runtime->create_physical_instance(ctx,
                                               target_memory,
                                               *layout_constraints,
                                               regions,
                                               *result,
                                               true /*acquire*/,
                                               LEGION_GC_DEFAULT_PRIORITY,
                                               false /*tight bounds*/,
                                               footprint);
    }

    auto created = false;
    // This needs to be find_or_create_physical_instance rather than create_physical_instance,
    // because multiple creation requests for the same region can reach here before the cache gets
    // updated.
    const auto success = runtime->find_or_create_physical_instance(ctx,
                                                                   target_memory,
                                                                   *layout_constraints,
                                                                   regions,
                                                                   *result,
                                                                   created,
                                                                   true /*acquire*/,
                                                                   LEGION_GC_DEFAULT_PRIORITY,
                                                                   false /*tight bounds*/,
                                                                   footprint);
    // The instance needs to be acquired only when it is found and not created
    *need_acquire = !created;
    return success;
  };

  if (!find_or_create_instance()) {
    return false;
  }

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    Realm::LoggerMessage msg = logger().debug();

    if (*need_acquire) {
      msg << "Operation " << mappable.get_unique_id() << ": found reduction instance " << result
          << " for";
      for (auto&& r : regions) {
        msg << " " << r;
      }
    } else {
      msg << "Operation " << mappable.get_unique_id() << ": created reduction instance " << result
          << " for";
      for (auto&& r : regions) {
        msg << " " << r;
      }
      msg << " (size: " << *footprint << " bytes, memory: " << target_memory << ")";
    }
  }

  // Don't need to do all the rest on instance reuse
  // *need_acquire == false means the instance is fresh
  if (*need_acquire) {
    return true;
  }

  if (can_cache) {
    // Record reduction instance
    reduction_instances_.record_instance(redop, regions.front(), fields.front(), *result, policy);
  }

  runtime->subscribe(ctx, *result);
  // Record the operation that created this instance
  if (const auto provenance = mappable.get_provenance_string(); !provenance.empty()) {
    creating_operation_[*result] = std::string{provenance};
  }

  return true;
}

bool BaseMapper::map_regular_instance_(const Legion::Mapping::MapperContext& ctx,
                                       const Legion::Mappable& mappable,
                                       const std::set<const Legion::RegionRequirement*>& reqs,
                                       const InstanceMappingPolicy& policy,
                                       const std::vector<Legion::FieldID>& fields,
                                       const Legion::LayoutConstraintSet& layout_constraints,
                                       Memory target_memory,
                                       bool must_alloc_collective_writes,
                                       std::vector<Legion::LogicalRegion>&& regions,
                                       Legion::Mapping::PhysicalInstance* result,
                                       bool* need_acquire,
                                       std::size_t* footprint)
{
  constexpr auto has_collective_write =
    [](const std::set<const Legion::RegionRequirement*>& to_check) {
      return std::any_of(
        to_check.begin(), to_check.end(), [](const Legion::RegionRequirement* req) {
          return ((req->privilege & LEGION_WRITE_PRIV) != 0) &&
                 ((req->prop & LEGION_COLLECTIVE_MASK) != 0);
        });
    };
  const auto alloc_policy = must_alloc_collective_writes && has_collective_write(reqs)
                              ? AllocPolicy::MUST_ALLOC
                              : policy.allocation;
  const auto can_cache    = fields.size() == 1 && regions.size() == 1 &&
                         alloc_policy != AllocPolicy::MUST_ALLOC &&
                         // Redundant copies get invalidated immediately after this operation so
                         // they shouldn't be cached
                         !policy.redundant;

  const auto found_in_cache = [&] {
    auto cached =
      local_instances_.find_instance(regions.front(), fields.front(), target_memory, policy);
    const auto found = cached.has_value();

    if (found) {
      *result = *std::move(cached);
      if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
        logger().debug() << "Operation " << mappable.get_unique_id() << ": reused cached instance "
                         << *result << " for " << regions.front();
      }
    }
    return found;
  };

  if (can_cache && found_in_cache()) {
    // Needs acquire if found to keep the runtime happy
    *need_acquire = true;
    return true /* success */;
  }

  const auto domain = [&]() -> std::optional<Domain> {
    if (fields.size() == 1 && regions.size() == 1) {
      // When the client mapper didn't request colocation and also didn't want the instance
      // to be exact, we can do an interesting optimization here to try to reduce unnecessary
      // inter-memory copies. For logical regions that are overlapping we try
      // to accumulate as many as possible into one physical instance and use
      // that instance for all the tasks for the different regions.
      // First we have to see if there is anything we overlap with
      auto is = regions.front().get_index_space();

      if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
        logger().debug() << "Operation " << mappable.get_unique_id()
                         << ": waiting on index space domain for " << is << " for "
                         << regions.front();
      }
      return runtime->get_index_space_domain(ctx, std::move(is));
    }
    return std::nullopt;
  }();

  InternalSharedPtr<RegionGroup> group{nullptr};

  if (domain.has_value()) {
    // During get_index_space_domain() it is possible (though unlikely) that another mapping
    // task pre-empted us and deposited a matching region into the cache.
    //
    // So we need to search the cache again, because otherwise we might end up with 2 instances
    // covering the same region in the cache.
    if (can_cache && found_in_cache()) {
      // Needs acquire if found to keep the runtime happy
      *need_acquire = true;
      return true /* success */;
    }
    group = local_instances_.find_region_group(
      regions.front(), *domain, fields.front(), target_memory, policy.exact);
    regions.assign(group->regions.begin(), group->regions.end());
  }

  // Haven't made this instance before, so make it now
  const auto find_or_create_instance = [&] {
    if (alloc_policy == AllocPolicy::MUST_ALLOC) {
      *need_acquire = false;
      return runtime->create_physical_instance(ctx,
                                               target_memory,
                                               layout_constraints,
                                               regions,
                                               *result,
                                               true /*acquire*/,
                                               LEGION_GC_DEFAULT_PRIORITY,
                                               policy.exact /*tight bounds*/,
                                               footprint);
    }

    bool created = true;
    // This needs to be find_or_create_physical_instance rather than create_physical_instance,
    // because multiple creation requests that result in the same physical instance can reach here
    // before the cache gets updated.
    const auto success = runtime->find_or_create_physical_instance(ctx,
                                                                   target_memory,
                                                                   layout_constraints,
                                                                   regions,
                                                                   *result,
                                                                   created,
                                                                   true /*acquire*/,
                                                                   LEGION_GC_DEFAULT_PRIORITY,
                                                                   policy.exact /*tight bounds*/,
                                                                   footprint);
    *need_acquire      = !created;
    return success;
  };

  if (!find_or_create_instance()) {
    if (group != nullptr) {
      LEGATE_CHECK(fields.size() == 1);
      local_instances_.remove_pending_instance(
        regions.front(), group, fields.front(), target_memory);
    }
    return false;
  }

  // We succeeded in making the instance where we want it
  LEGATE_CHECK(result->exists());
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    // *need_acquire == false means the instance is fresh
    auto&& mess            = logger().debug();
    const auto print_group = [&] {
      if (group) {
        mess << *group;
      } else {
        mess << "(no RegionGroup)";
      }
      // We return a dummy empty string here so that this function (when called) can be
      // pipelined with other << commands.
      return "";
    };

    if (*need_acquire) {
      mess << "Operation " << mappable.get_unique_id() << ": found instance " << *result << " for "
           << print_group();
    } else {
      mess << "Operation " << mappable.get_unique_id() << ": created instance " << *result
           << " for " << print_group() << " (size: " << std::dec << *footprint
           << " bytes, memory: " << std::hex << target_memory << ")";
    }
  }
  if (group != nullptr) {
    LEGATE_CHECK(fields.size() == 1);
    local_instances_.record_instance(regions.front(), group, fields.front(), *result, policy);
  }
  // *need_acquire == false means the instance is fresh
  if (!*need_acquire) {
    runtime->subscribe(ctx, *result);
    // Record the operation that created this instance
    if (const auto provenance = mappable.get_provenance_string(); !provenance.empty()) {
      creating_operation_[*result] = std::string{provenance};
    }
  }

  return true;
}

bool BaseMapper::map_legate_store_(Legion::Mapping::MapperContext ctx,
                                   const Legion::Mappable& mappable,
                                   const StoreMapping& mapping,
                                   const std::set<const Legion::RegionRequirement*>& reqs,
                                   Processor target_proc,
                                   Legion::Mapping::PhysicalInstance& result,
                                   bool can_fail,
                                   bool must_alloc_collective_writes)
{
  if (reqs.empty()) {
    return false;
  }

  auto redop = GlobalRedopID{(*reqs.begin())->redop};
  std::vector<Legion::LogicalRegion> regions;

  regions.reserve(reqs.size());
  for (const auto* req : reqs) {
    // Colocated stores should be either non-reduction arguments or reductions with the same
    // reduction operator.
    LEGATE_ASSERT(redop == GlobalRedopID{req->redop});
    if (LEGION_NO_ACCESS == req->privilege) {
      continue;
    }
    regions.push_back(req->region);
  }

  if (regions.empty()) {
    return false;
  }

  // Targets of reduction copies should be mapped to normal instances
  if (mappable.get_mappable_type() == Legion::Mappable::COPY_MAPPABLE) {
    redop = GlobalRedopID{0};
  }

  const auto& policy       = mapping.policy;
  const auto target_memory = local_machine_.get_memory(target_proc, policy.target);

  // Generate layout constraints from the store mapping
  Legion::LayoutConstraintSet layout_constraints;

  mapping.populate_layout_constraints(layout_constraints);

  const auto& fields    = layout_constraints.field_constraint.field_set;
  bool need_acquire     = false;
  std::size_t footprint = 0;
  bool success          = false;

  if (redop == GlobalRedopID{0}) {
    success = map_regular_instance_(ctx,
                                    mappable,
                                    reqs,
                                    policy,
                                    fields,
                                    layout_constraints,
                                    target_memory,
                                    must_alloc_collective_writes,
                                    std::move(regions),
                                    &result,
                                    &need_acquire,
                                    &footprint);
  } else {
    success = map_reduction_instance_(ctx,
                                      mappable,
                                      target_proc,
                                      fields,
                                      regions,
                                      policy,
                                      target_memory,
                                      redop,
                                      &layout_constraints,
                                      &result,
                                      &need_acquire,
                                      &footprint);
  }

  if (success) {
    return need_acquire;
  }
  // If we make it here then we failed entirely
  if (!can_fail) {
    report_failed_mapping_(ctx, mappable, mapping, target_memory, redop, footprint);
  }
  return true;
}

void BaseMapper::report_failed_mapping_(Legion::Mapping::MapperContext ctx,
                                        const Legion::Mappable& mappable,
                                        const StoreMapping& mapping,
                                        Memory target_memory,
                                        GlobalRedopID redop,
                                        std::size_t footprint)
{
  std::string_view opname;
  if (mappable.get_mappable_type() == Legion::Mappable::TASK_MAPPABLE) {
    opname = mappable.as_task()->get_task_name();
  }

  std::string_view provenance = mappable.get_provenance_string();
  if (provenance.empty()) {
    provenance = "unknown provenance";
  }

  std::stringstream req_ss;

  if (redop > GlobalRedopID{0}) {
    req_ss << "reduction (" << traits::detail::to_underlying(redop) << ") requirement(s) ";
  } else {
    req_ss << "region requirement(s) ";
  }
  req_ss << fmt::format("{}", mapping.requirement_indices());

  logger().error() << "Failed to allocate " << footprint << " bytes on memory " << std::hex
                   << target_memory.id << std::dec << " (of kind "
                   << Legion::Mapping::Utilities::to_string(target_memory.kind()) << ") for "
                   << req_ss.str() << " of " << log_mappable(mappable, true /*prefix_only*/)
                   << opname << "[" << provenance << "] (UID " << mappable.get_unique_id() << ")";
  for (const Legion::RegionRequirement* req : mapping.requirements()) {
    for (const Legion::FieldID fid : req->instance_fields) {
      logger().error() << "  corresponding to a LogicalStore allocated at "
                       << retrieve_alloc_info_(ctx, req->region.get_field_space(), fid);
    }
  }

  std::vector<Legion::Mapping::PhysicalInstance> existing;

  runtime->find_physical_instances(ctx,
                                   target_memory,
                                   Legion::LayoutConstraintSet{},
                                   std::vector<Legion::LogicalRegion>{},
                                   existing);

  using StoreKey = std::pair<Legion::FieldSpace, Legion::FieldID>;
  std::unordered_map<StoreKey, std::vector<Legion::Mapping::PhysicalInstance>, hasher<StoreKey>>
    insts_for_store{};
  std::size_t total_size = 0;

  for (const Legion::Mapping::PhysicalInstance& inst : existing) {
    std::set<Legion::FieldID> fields;

    total_size += inst.get_instance_size();
    inst.get_fields(fields);
    for (const Legion::FieldID fid : fields) {
      insts_for_store[{inst.get_field_space(), fid}].push_back(inst);
    }
  }

  // TODO(mpapadakis): Once the one-pool solution is merged, we will no longer need to mention eager
  // allocations here, but will have to properly report the case where a task reserving an eager
  // instance (of known size) has not finished yet, so its reserved eager memory has not yet been
  // returned to the deferred pool.
  //
  // UPDATE(wonchanl): Removing the mentioning of the eager pool from the message, but the reporting
  // still needs to be extended per the description above.
  logger().error() << "There is not enough space because Legate is reserving " << total_size
                   << " of the available " << target_memory.capacity()
                   << " bytes for the following LogicalStores:";
  for (auto&& [store_key, insts] : insts_for_store) {
    auto&& [fs, fid] = store_key;

    logger().error() << "LogicalStore allocated at " << retrieve_alloc_info_(ctx, fs, fid) << ":";
    for (const Legion::Mapping::PhysicalInstance& inst : insts) {
      std::set<Legion::FieldID> fields;
      inst.get_fields(fields);
      logger().error() << "  Instance " << std::hex << inst.get_instance_id() << std::dec
                       << " of size " << inst.get_instance_size() << " covering elements "
                       << Legion::Mapping::Utilities::to_string(
                            runtime, ctx, inst.get_instance_domain())
                       << (fields.size() > 1 ? " of multiple stores" : " ");
      if (const auto it = creating_operation_.find(inst); it != creating_operation_.end()) {
        logger().error() << "    created for an operation launched at " << it->second;
      }
    }
  }

  LEGATE_ABORT("Out of memory");
}

void BaseMapper::select_task_variant(Legion::Mapping::MapperContext ctx,
                                     const Legion::Task& task,
                                     const SelectVariantInput& input,
                                     SelectVariantOutput& output)
{
  output.chosen_variant = select_task_variant_(ctx, task, input.processor);
}

void BaseMapper::postmap_task(Legion::Mapping::MapperContext /*ctx*/,
                              const Legion::Task& /*task*/,
                              const PostMapInput& /*input*/,
                              PostMapOutput& /*output*/)
{
  // We should currently never get this call in Legate
  LEGATE_ABORT("Should never get here");
}

void BaseMapper::select_task_sources(Legion::Mapping::MapperContext ctx,
                                     const Legion::Task& /*task*/,
                                     const SelectTaskSrcInput& input,
                                     SelectTaskSrcOutput& output)
{
  legate_select_sources_(
    ctx, input.target, input.source_instances, input.collective_views, output.chosen_ranking);
}

namespace {

using Bandwidth = std::uint32_t;

// Source instance annotated with the memory bandwidth
class AnnotatedSourceInstance {
 public:
  AnnotatedSourceInstance(Legion::Mapping::PhysicalInstance _instance, Bandwidth _bandwidth)
    : instance{std::move(_instance)}, bandwidth{_bandwidth}
  {
  }

  Legion::Mapping::PhysicalInstance instance{};
  Bandwidth bandwidth{};
};

[[nodiscard]] Bandwidth compute_bandwidth(const Legion::Machine& legion_machine,
                                          const Realm::Memory& source_memory,
                                          const Realm::Memory& target_memory,
                                          const LocalMachine& local_machine)
{
  std::vector<Legion::MemoryMemoryAffinity> affinities;

  legion_machine.get_mem_mem_affinity(
    affinities, source_memory, target_memory, false /*not just local affinities*/);
  // affinities being empty means that there's no direct channel between the source and target
  // memories, in which case we assign the smallest bandwidth
  // TODO(wonchanl): Not all multi-hop copies are equal
  if (!affinities.empty()) {
    LEGATE_ASSERT(affinities.size() == 1);
    return affinities.front().bandwidth;
  }

  if (source_memory.kind() == Realm::Memory::GPU_FB_MEM) {
    // Last resort: check if this is a special case of a local-node multi-hop CPU<->GPU copy.
    return local_machine.g2c_multi_hop_bandwidth(source_memory, target_memory);
  }

  if (target_memory.kind() == Realm::Memory::GPU_FB_MEM) {
    // Symmetric case to the above
    return local_machine.g2c_multi_hop_bandwidth(target_memory, source_memory);
  }

  return Bandwidth{0};
}

void find_source_instance_bandwidth(const Legion::Mapping::PhysicalInstance& source_instance,
                                    const Memory& target_memory,
                                    const Legion::Machine& legion_machine,
                                    const LocalMachine& local_machine,
                                    std::vector<AnnotatedSourceInstance>* all_sources,
                                    std::unordered_map<Memory, Bandwidth>* source_memory_bandwidths)
{
  const auto bandwidth = [&] {
    const auto source_memory  = source_instance.get_location();
    const auto [it, inserted] = source_memory_bandwidths->try_emplace(source_memory);

    if (inserted) {
      // Haven't seen this memory before
      it->second = compute_bandwidth(legion_machine, source_memory, target_memory, local_machine);
    }
    return it->second;
  }();

  all_sources->emplace_back(source_instance, bandwidth);
}

}  // namespace

void BaseMapper::legate_select_sources_(
  Legion::Mapping::MapperContext ctx,
  const Legion::Mapping::PhysicalInstance& target,
  const std::vector<Legion::Mapping::PhysicalInstance>& sources,
  const std::vector<Legion::Mapping::CollectiveView>& collective_sources,
  std::deque<Legion::Mapping::PhysicalInstance>& ranking)
{
  std::unordered_map<Memory, Bandwidth> source_memory_bandwidths;
  // For right now we'll rank instances by the bandwidth of the memory
  // they are in to the destination.
  // TODO(wonchanl): consider layouts when ranking source to help out the DMA system
  auto target_memory = target.get_location();
  // fill in a vector of the sources with their bandwidths
  std::vector<AnnotatedSourceInstance> all_sources;

  all_sources.reserve(sources.size() + collective_sources.size());
  for (auto&& source : sources) {
    find_source_instance_bandwidth(source,
                                   target_memory,
                                   legion_machine_,
                                   local_machine_,
                                   &all_sources,
                                   &source_memory_bandwidths);
  }

  for (auto&& collective_source : collective_sources) {
    std::vector<Legion::Mapping::PhysicalInstance> source_instances;

    collective_source.find_instances_nearest_memory(target_memory, source_instances);
    // there must exist at least one instance in the collective view
    LEGATE_ASSERT(!source_instances.empty());
    // we need only first instance if there are several
    find_source_instance_bandwidth(source_instances.front(),
                                   target_memory,
                                   legion_machine_,
                                   local_machine_,
                                   &all_sources,
                                   &source_memory_bandwidths);
  }
  LEGATE_ASSERT(!all_sources.empty());
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    logger().debug() << "Selecting sources for "
                     << Legion::Mapping::Utilities::to_string(runtime, ctx, target);
    for (auto&& [i, src] : legate::detail::enumerate(all_sources)) {
      logger().debug() << ((i < static_cast<std::int64_t>(sources.size())) ? "Standalone"
                                                                           : "Collective")
                       << " source "
                       << Legion::Mapping::Utilities::to_string(runtime, ctx, src.instance)
                       << " bandwidth " << src.bandwidth;
    }
  }

  // Sort source instances by their bandwidths
  std::sort(all_sources.begin(), all_sources.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.bandwidth > rhs.bandwidth;
  });
  // Record all instances from the one of the largest bandwidth to that of the smallest
  for (auto&& source : all_sources) {
    ranking.emplace_back(source.instance);
  }
}

void BaseMapper::report_profiling(Legion::Mapping::MapperContext,
                                  const Legion::Task&,
                                  const TaskProfilingInfo&)
{
  LEGATE_ABORT("Shouldn't get any profiling feedback currently");
}

Legion::ShardingID BaseMapper::find_mappable_sharding_functor_id_(const Legion::Mappable& mappable)
{
  const Mappable legate_mappable{&mappable};

  return static_cast<Legion::ShardingID>(legate_mappable.sharding_id());
}

void BaseMapper::select_sharding_functor(Legion::Mapping::MapperContext,
                                         const Legion::Task& task,
                                         const SelectShardingFunctorInput&,
                                         SelectShardingFunctorOutput& output)
{
  output.chosen_functor = find_mappable_sharding_functor_id_(task);
}

void BaseMapper::map_inline(Legion::Mapping::MapperContext ctx,
                            const Legion::InlineMapping& inline_op,
                            const MapInlineInput& /*input*/,
                            MapInlineOutput& output)
{
  auto target_proc = [&] {
    if (local_machine_.has_omps()) {
      return local_machine_.omps().front();
    }
    return local_machine_.cpus().front();
  }();

  auto store_target = default_store_targets(target_proc.kind()).front();

  LEGATE_ASSERT(inline_op.requirement.instance_fields.size() == 1);

  const Store store{runtime, ctx, &inline_op.requirement};
  std::vector<std::unique_ptr<StoreMapping>> mappings;

  auto&& reqs = mappings.emplace_back(StoreMapping::default_mapping(&store, store_target, false))
                  ->requirements();

  OutputMap output_map;

  output_map.reserve(reqs.size());
  for (auto* req : reqs) {
    output_map[req] = &output.chosen_instances;
  }

  map_legate_stores_(ctx, inline_op, mappings, target_proc, output_map);
}

void BaseMapper::select_inline_sources(Legion::Mapping::MapperContext ctx,
                                       const Legion::InlineMapping& /*inline_op*/,
                                       const SelectInlineSrcInput& input,
                                       SelectInlineSrcOutput& output)
{
  legate_select_sources_(
    ctx, input.target, input.source_instances, input.collective_views, output.chosen_ranking);
}

void BaseMapper::report_profiling(Legion::Mapping::MapperContext /*ctx*/,
                                  const Legion::InlineMapping& /*inline_op*/,
                                  const InlineProfilingInfo& /*input*/)
{
  LEGATE_ABORT("No profiling yet for inline mappings");
}

void BaseMapper::map_copy(Legion::Mapping::MapperContext ctx,
                          const Legion::Copy& copy,
                          const MapCopyInput& /*input*/,
                          MapCopyOutput& output)
{
  const Copy legate_copy{&copy, runtime, ctx};
  output.copy_fill_priority = legate_copy.priority();

  auto& machine_desc = legate_copy.machine();
  auto copy_target   = [&]() {
    // If we're mapping an indirect copy and have data resident in GPU memory,
    // map everything to CPU memory, as indirect copies on GPUs are currently
    // extremely slow.
    auto indirect =
      !copy.src_indirect_requirements.empty() || !copy.dst_indirect_requirements.empty();
    auto&& valid_targets = indirect ? machine_desc.valid_targets_except({TaskTarget::GPU})
                                      : machine_desc.valid_targets();
    // However, if the machine in the scope doesn't have any CPU or OMP as a fallback for
    // indirect copies, we have no choice but using GPUs
    if (valid_targets.empty()) {
      LEGATE_ASSERT(indirect);
      return machine_desc.valid_targets().front();
    }
    return valid_targets.front();
  }();

  auto local_range = local_machine_.slice(copy_target, machine_desc, true);
  Processor target_proc;
  if (copy.is_index_space) {
    Domain sharding_domain = copy.index_domain;
    if (copy.sharding_space.exists()) {
      sharding_domain = runtime->get_index_space_domain(ctx, copy.sharding_space);
    }

    // FIXME: We might later have non-identity projections for copy requirements,
    // in which case we should find the key store and use its projection functor
    // for the linearization
    auto* key_functor = legate::detail::find_projection_function(0);
    auto lo           = key_functor->project_point(sharding_domain.lo());
    auto hi           = key_functor->project_point(sharding_domain.hi());
    auto p            = key_functor->project_point(copy.index_point);

    const std::uint32_t start_proc_id     = machine_desc.processor_range().low;
    const std::uint32_t total_tasks_count = legate::detail::linearize(lo, hi, hi) + 1;
    auto idx =
      (legate::detail::linearize(lo, hi, p) * local_range.total_proc_count() / total_tasks_count) +
      start_proc_id;
    target_proc = local_range[idx];
  } else {
    target_proc = local_range.first();
  }

  auto store_target = default_store_targets(target_proc.kind()).front();

  OutputMap output_map;
  auto add_to_output_map =
    [&output_map](const std::vector<Legion::RegionRequirement>& reqs,
                  std::vector<std::vector<Legion::Mapping::PhysicalInstance>>& instances) {
      instances.resize(reqs.size());
      for (auto&& [req, inst] : legate::detail::zip_equal(reqs, instances)) {
        output_map[&req] = &inst;
      }
    };
  add_to_output_map(copy.src_requirements, output.src_instances);
  add_to_output_map(copy.dst_requirements, output.dst_instances);

  LEGATE_ASSERT(copy.src_indirect_requirements.size() <= 1);
  LEGATE_ASSERT(copy.dst_indirect_requirements.size() <= 1);
  if (!copy.src_indirect_requirements.empty()) {
    // This is to make the push_back call later add the instance to the right place
    output.src_indirect_instances.clear();
    output_map[&copy.src_indirect_requirements.front()] = &output.src_indirect_instances;
  }
  if (!copy.dst_indirect_requirements.empty()) {
    // This is to make the push_back call later add the instance to the right place
    output.dst_indirect_instances.clear();
    output_map[&copy.dst_indirect_requirements.front()] = &output.dst_indirect_instances;
  }

  auto&& inputs     = legate_copy.inputs();
  auto&& outputs    = legate_copy.outputs();
  auto&& input_ind  = legate_copy.input_indirections();
  auto&& output_ind = legate_copy.output_indirections();

  const auto stores_to_copy = {
    std::ref(inputs), std::ref(outputs), std::ref(input_ind), std::ref(output_ind)};

  std::size_t reserve_size = 0;
  for (auto&& store : stores_to_copy) {
    reserve_size += store.get().size();
  }

  std::vector<std::unique_ptr<StoreMapping>> mappings;

  mappings.reserve(reserve_size);
  for (auto&& store_set : stores_to_copy) {
    for (auto&& store : store_set.get()) {
      mappings.emplace_back(StoreMapping::default_mapping(&store, store_target, false));
    }
  }
  map_legate_stores_(ctx, copy, mappings, target_proc, output_map);
}

void BaseMapper::select_copy_sources(Legion::Mapping::MapperContext ctx,
                                     const Legion::Copy& /*copy*/,
                                     const SelectCopySrcInput& input,
                                     SelectCopySrcOutput& output)
{
  legate_select_sources_(
    ctx, input.target, input.source_instances, input.collective_views, output.chosen_ranking);
}

void BaseMapper::report_profiling(Legion::Mapping::MapperContext /*ctx*/,
                                  const Legion::Copy& /*copy*/,
                                  const CopyProfilingInfo& /*input*/)
{
  LEGATE_ABORT("No profiling for copies yet");
}

void BaseMapper::select_sharding_functor(Legion::Mapping::MapperContext /*ctx*/,
                                         const Legion::Copy& copy,
                                         const SelectShardingFunctorInput& /*input*/,
                                         SelectShardingFunctorOutput& output)
{
  // TODO(wonchanl): Copies can have key stores in the future
  output.chosen_functor = find_mappable_sharding_functor_id_(copy);
}

void BaseMapper::select_close_sources(Legion::Mapping::MapperContext ctx,
                                      const Legion::Close& /*close*/,
                                      const SelectCloseSrcInput& input,
                                      SelectCloseSrcOutput& output)
{
  legate_select_sources_(
    ctx, input.target, input.source_instances, input.collective_views, output.chosen_ranking);
}

void BaseMapper::report_profiling(Legion::Mapping::MapperContext /*ctx*/,
                                  const Legion::Close& /*close*/,
                                  const CloseProfilingInfo& /*input*/)
{
  LEGATE_ABORT("No profiling yet for legate");
}

void BaseMapper::select_sharding_functor(Legion::Mapping::MapperContext /*ctx*/,
                                         const Legion::Close& /*close*/,
                                         const SelectShardingFunctorInput& /*input*/,
                                         SelectShardingFunctorOutput& /*output*/)
{
  LEGATE_ABORT("Should never get here");
}

void BaseMapper::map_acquire(Legion::Mapping::MapperContext /*ctx*/,
                             const Legion::Acquire& /*acquire*/,
                             const MapAcquireInput& /*input*/,
                             MapAcquireOutput& /*output*/)
{
  // Nothing to do
}

void BaseMapper::report_profiling(Legion::Mapping::MapperContext /*ctx*/,
                                  const Legion::Acquire& /*acquire*/,
                                  const AcquireProfilingInfo& /*input*/)
{
  LEGATE_ABORT("Should never get here");
}

void BaseMapper::select_sharding_functor(Legion::Mapping::MapperContext /*ctx*/,
                                         const Legion::Acquire& /*acquire*/,
                                         const SelectShardingFunctorInput& /*input*/,
                                         SelectShardingFunctorOutput& /*output*/)
{
  LEGATE_ABORT("Should never get here");
}

void BaseMapper::map_release(Legion::Mapping::MapperContext /*ctx*/,
                             const Legion::Release& /*release*/,
                             const MapReleaseInput& /*input*/,
                             MapReleaseOutput& /*output*/)
{
  // Nothing to do
}

void BaseMapper::select_release_sources(Legion::Mapping::MapperContext ctx,
                                        const Legion::Release& /*release*/,
                                        const SelectReleaseSrcInput& input,
                                        SelectReleaseSrcOutput& output)
{
  legate_select_sources_(
    ctx, input.target, input.source_instances, input.collective_views, output.chosen_ranking);
}

void BaseMapper::report_profiling(Legion::Mapping::MapperContext /*ctx*/,
                                  const Legion::Release& /*release*/,
                                  const ReleaseProfilingInfo& /*input*/)
{
  // No profiling for legate yet
  LEGATE_ABORT("No profiling for legate yet");
}

void BaseMapper::select_sharding_functor(Legion::Mapping::MapperContext /*ctx*/,
                                         const Legion::Release& /*release*/,
                                         const SelectShardingFunctorInput& /*input*/,
                                         SelectShardingFunctorOutput& /*output*/)
{
  LEGATE_ABORT("Should never get here");
}

void BaseMapper::select_partition_projection(Legion::Mapping::MapperContext /*ctx*/,
                                             const Legion::Partition& /*partition*/,
                                             const SelectPartitionProjectionInput& input,
                                             SelectPartitionProjectionOutput& output)
{
  // If we have an open complete partition then use it
  if (!input.open_complete_partitions.empty()) {
    output.chosen_partition = input.open_complete_partitions.front();
  } else {
    output.chosen_partition = Legion::LogicalPartition::NO_PART;
  }
}

void BaseMapper::map_partition(Legion::Mapping::MapperContext ctx,
                               const Legion::Partition& partition,
                               const MapPartitionInput&,
                               MapPartitionOutput& output)
{
  auto target_proc = [&] {
    if (local_machine_.has_omps()) {
      return local_machine_.omps().front();
    }
    return local_machine_.cpus().front();
  }();

  auto store_target = default_store_targets(target_proc.kind()).front();

  LEGATE_ASSERT(partition.requirement.instance_fields.size() == 1);

  const Store store{runtime, ctx, &partition.requirement};
  std::vector<std::unique_ptr<StoreMapping>> mappings;

  auto&& reqs = mappings.emplace_back(StoreMapping::default_mapping(&store, store_target, false))
                  ->requirements();

  OutputMap output_map;

  output_map.reserve(reqs.size());
  for (auto* req : reqs) {
    output_map[req] = &output.chosen_instances;
  }

  map_legate_stores_(ctx, partition, mappings, std::move(target_proc), output_map);
}

void BaseMapper::select_partition_sources(Legion::Mapping::MapperContext ctx,
                                          const Legion::Partition& /*partition*/,
                                          const SelectPartitionSrcInput& input,
                                          SelectPartitionSrcOutput& output)
{
  legate_select_sources_(
    ctx, input.target, input.source_instances, input.collective_views, output.chosen_ranking);
}

void BaseMapper::report_profiling(Legion::Mapping::MapperContext /*ctx*/,
                                  const Legion::Partition& /*partition*/,
                                  const PartitionProfilingInfo& /*input*/)
{
  // No profiling yet
  LEGATE_ABORT("No profiling for partition ops yet");
}

void BaseMapper::select_sharding_functor(Legion::Mapping::MapperContext /*ctx*/,
                                         const Legion::Partition& partition,
                                         const SelectShardingFunctorInput& /*input*/,
                                         SelectShardingFunctorOutput& output)
{
  output.chosen_functor = find_mappable_sharding_functor_id_(partition);
}

void BaseMapper::select_sharding_functor(Legion::Mapping::MapperContext /*ctx*/,
                                         const Legion::Fill& fill,
                                         const SelectShardingFunctorInput& /*input*/,
                                         SelectShardingFunctorOutput& output)
{
  output.chosen_functor = find_mappable_sharding_functor_id_(fill);
}

void BaseMapper::configure_context(Legion::Mapping::MapperContext /*ctx*/,
                                   const Legion::Task& /*task*/,
                                   ContextConfigOutput& /*output*/)
{
  // Use the defaults currently
}

void BaseMapper::map_future_map_reduction(Legion::Mapping::MapperContext /*ctx*/,
                                          const FutureMapReductionInput& input,
                                          FutureMapReductionOutput& output)
{
  output.serdez_upper_bound = LEGATE_MAX_SIZE_SCALAR_RETURN;
  auto& dest_memories       = output.destination_memories;

  if (local_machine_.has_gpus()) {
    // TODO(wonchanl): It's been reported that blindly mapping target instances of future map
    // reductions to framebuffers hurts performance. Until we find a better mapping policy, we guard
    // the current policy with a macro.
    if (LEGATE_DEFINED(LEGATE_MAP_FUTURE_MAP_REDUCTIONS_TO_GPU)) {
      // If this was joining exceptions, we should put instances on a host-visible memory
      // because they need serdez
      if (input.tag ==
          traits::detail::to_underlying(legate::detail::CoreMappingTag::JOIN_EXCEPTION)) {
        dest_memories.push_back(local_machine_.zerocopy_memory());
      } else {
        auto&& fbufs = local_machine_.frame_buffers();

        dest_memories.reserve(fbufs.size());
        for (auto&& pair : fbufs) {
          dest_memories.push_back(pair.second);
        }
      }
    } else {
      dest_memories.push_back(local_machine_.zerocopy_memory());
    }
  } else if (local_machine_.has_socket_memory()) {
    auto&& smems = local_machine_.socket_memories();

    dest_memories.reserve(smems.size());
    for (auto&& pair : smems) {
      dest_memories.push_back(pair.second);
    }
  }
}

void BaseMapper::select_tunable_value(Legion::Mapping::MapperContext /*ctx*/,
                                      const Legion::Task& /*task*/,
                                      const SelectTunableInput& input,
                                      SelectTunableOutput& output)
{
  auto* user_mapper = const_cast<mapping::Mapper*>(static_cast<const mapping::Mapper*>(input.args));
  const auto value  = user_mapper->tunable_value(input.tunable_id);
  const auto size   = value.size();

  output.size = size;
  if (size) {
    output.value = std::malloc(size);
    LEGATE_ASSERT(output.value);
    std::memcpy(output.value, value.ptr(), size);
  } else {
    output.value = nullptr;
  }
}

void BaseMapper::select_sharding_functor(Legion::Mapping::MapperContext /*ctx*/,
                                         const Legion::MustEpoch& /*epoch*/,
                                         const SelectShardingFunctorInput& /*input*/,
                                         MustEpochShardingFunctorOutput& /*output*/)
{
  LEGATE_ABORT("Should never get here");
}

void BaseMapper::memoize_operation(Legion::Mapping::MapperContext /*ctx*/,
                                   const Legion::Mappable& /*mappable*/,
                                   const MemoizeInput& /*input*/,
                                   MemoizeOutput& output)
{
  output.memoize = true;
}

void BaseMapper::map_must_epoch(Legion::Mapping::MapperContext /*ctx*/,
                                const MapMustEpochInput& /*input*/,
                                MapMustEpochOutput& /*output*/)
{
  LEGATE_ABORT("Should never get here");
}

void BaseMapper::map_dataflow_graph(Legion::Mapping::MapperContext /*ctx*/,
                                    const MapDataflowGraphInput& /*input*/,
                                    MapDataflowGraphOutput& /*output*/)
{
  LEGATE_ABORT("Should never get here");
}

void BaseMapper::select_tasks_to_map(Legion::Mapping::MapperContext /*ctx*/,
                                     const SelectMappingInput& input,
                                     SelectMappingOutput& output)
{
  // Just map all the ready tasks
  output.map_tasks.insert(input.ready_tasks.begin(), input.ready_tasks.end());
}

void BaseMapper::select_steal_targets(Legion::Mapping::MapperContext /*ctx*/,
                                      const SelectStealingInput& /*input*/,
                                      SelectStealingOutput& /*output*/)
{
  // Nothing to do, no stealing in the leagte mapper currently
}

void BaseMapper::permit_steal_request(Legion::Mapping::MapperContext /*ctx*/,
                                      const StealRequestInput& /*input*/,
                                      StealRequestOutput& /*output*/)
{
  LEGATE_ABORT("no stealing in the legate mapper currently");
}

void BaseMapper::handle_message(Legion::Mapping::MapperContext /*ctx*/,
                                const MapperMessage& /*message*/)
{
  LEGATE_ABORT("We shouldn't be receiving any messages currently");
}

void BaseMapper::handle_task_result(Legion::Mapping::MapperContext /*ctx*/,
                                    const MapperTaskResult& /*result*/)
{
  LEGATE_ABORT("Nothing to do since we should never get one of these");
}

void BaseMapper::handle_instance_collection(Legion::Mapping::MapperContext /*ctx*/,
                                            const Legion::Mapping::PhysicalInstance& inst)
{
  if (!local_instances_.erase(inst)) {
    static_cast<void>(reduction_instances_.erase(inst));
    // It's OK if neither erase succeeds. This indicates that the instance was deleted in an
    // earlier call to record_instance() in which several instances were combined.
    //
    // Probably we should provide some mechanism to allow unsubscription in that case, but
    // that would be complicated to implement.
  }
  creating_operation_.erase(inst);
}

std::string_view BaseMapper::retrieve_alloc_info_(Legion::Mapping::MapperContext ctx,
                                                  Legion::FieldSpace fs,
                                                  Legion::FieldID fid)
{
  constexpr auto tag =
    static_cast<Legion::SemanticTag>(legate::detail::CoreSemanticTag::ALLOC_INFO);
  const void* orig_info;
  std::size_t size;

  if (runtime->retrieve_semantic_information(ctx,
                                             fs,
                                             fid,
                                             tag,
                                             orig_info,
                                             size,
                                             /*can_fail=*/true)) {
    const char* alloc_info = static_cast<const char*>(orig_info);

    if (size > 0 && alloc_info[0] != '\0') {
      return {alloc_info, size};
    }
  }
  return "(unknown provenance)";
}

}  // namespace legate::mapping::detail
