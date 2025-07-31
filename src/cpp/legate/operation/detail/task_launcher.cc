/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/task_launcher.h>

#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/launcher_arg.h>
#include <legate/operation/detail/store_analyzer.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/partition_manager.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/task/detail/returned_exception.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/small_vector.h>

#include <cstdint>
#include <optional>
#include <utility>

namespace legate::detail {

GlobalTaskID TaskLauncher::legion_task_id() const { return library_.get().get_task_id(task_id_); }

void TaskLauncher::reserve_inputs(std::size_t num) { inputs_.reserve(num); }

void TaskLauncher::add_input(Analyzable arg) { inputs_.push_back(std::move(arg)); }

void TaskLauncher::reserve_outputs(std::size_t num) { outputs_.reserve(num); }

void TaskLauncher::add_output(Analyzable arg) { outputs_.push_back(std::move(arg)); }

void TaskLauncher::reserve_reductions(std::size_t num) { reductions_.reserve(num); }

void TaskLauncher::add_reduction(Analyzable arg) { reductions_.push_back(std::move(arg)); }

void TaskLauncher::reserve_scalars(std::size_t num) { scalars_.reserve(num); }

void TaskLauncher::add_scalar(InternalSharedPtr<Scalar> scalar)
{
  scalars_.emplace_back(std::move(scalar));
}

void TaskLauncher::add_future(Legion::Future future) { futures_.push_back(std::move(future)); }

void TaskLauncher::add_future_map(Legion::FutureMap future_map)
{
  future_maps_.push_back(std::move(future_map));
}

void TaskLauncher::set_concurrent(bool is_concurrent)
{
  if (is_concurrent && parallel_policy().streaming()) {
    throw TracedException<std::runtime_error>{
      "Concurrent Tasks are not allowed inside a Streaming Scope. Please place the concurrent task "
      "outside the Streaming Scope."};
  }

  concurrent_ = is_concurrent;
}

void TaskLauncher::reserve_communicators(std::size_t num) { communicators_.reserve(num); }

void TaskLauncher::add_communicator(Legion::FutureMap communicator)
{
  communicators_.push_back(std::move(communicator));
  set_concurrent(true);
}

Legion::FutureMap TaskLauncher::execute(const Legion::Domain& launch_domain)
{
  StoreAnalyzer analyzer;

  analyze_arguments_(/* parallel */ true, &analyzer);

  auto task_arg = pack_task_arg_(/* parallel */ true, &analyzer);

  BufferBuilder mapper_arg;

  pack_mapper_arg_(mapper_arg);

  auto&& runtime = Runtime::get_runtime();

  Legion::IndexTaskLauncher index_task{static_cast<Legion::TaskID>(legion_task_id()),
                                       launch_domain,
                                       task_arg.to_legion_buffer(),
                                       Legion::ArgumentMap(),
                                       Legion::Predicate::TRUE_PRED,
                                       false /*must*/,
                                       runtime.mapper_id(),
                                       tag_,
                                       mapper_arg.to_legion_buffer()};

  index_task.provenance         = provenance().as_string_view();
  index_task.future_return_size = get_future_size_including_exception_();

  std::vector<Legion::OutputRequirement> output_requirements;

  analyzer.populate(index_task, output_requirements);

  if (insert_barrier_) {
    const auto num_tasks                 = launch_domain.get_volume();
    auto [arrival_barrier, wait_barrier] = runtime.create_barriers(num_tasks);

    index_task.add_future(Legion::Future::from_value(arrival_barrier));
    index_task.add_future(Legion::Future::from_value(wait_barrier));
    runtime.destroy_barrier(arrival_barrier);
    runtime.destroy_barrier(wait_barrier);
  }
  for (auto&& communicator : communicators_) {
    index_task.point_futures.emplace_back(communicator);
  }

  index_task.concurrent = concurrent_;

  auto result = runtime.dispatch(index_task, output_requirements);

  post_process_unbound_stores_(result, launch_domain, output_requirements);
  for (auto&& arg : outputs_) {
    std::visit([](auto&& a) { a.perform_invalidations(); }, arg);
  }
  return result;
}

Legion::Future TaskLauncher::execute_single()
{
  StoreAnalyzer analyzer;

  analyze_arguments_(/* parallel */ false, &analyzer);

  auto task_arg = pack_task_arg_(/* parallel */ false, &analyzer);

  BufferBuilder mapper_arg;

  pack_mapper_arg_(mapper_arg);

  auto&& runtime = Runtime::get_runtime();

  Legion::TaskLauncher single_task{static_cast<Legion::TaskID>(legion_task_id()),
                                   task_arg.to_legion_buffer(),
                                   Legion::Predicate::TRUE_PRED,
                                   runtime.mapper_id(),
                                   tag_,
                                   mapper_arg.to_legion_buffer()};

  single_task.provenance         = provenance().as_string_view();
  single_task.future_return_size = get_future_size_including_exception_();

  std::vector<Legion::OutputRequirement> output_requirements;

  analyzer.populate(single_task, output_requirements);

  single_task.local_function_task = !has_side_effect_ && analyzer.can_be_local_function_task();

  auto result = runtime.dispatch(single_task, output_requirements);
  post_process_unbound_stores_(output_requirements);
  for (auto&& arg : outputs_) {
    std::visit([](auto&& a) { a.perform_invalidations(); }, arg);
  }
  return result;
}

namespace {

void analyze(Span<const Analyzable> args, StoreAnalyzer& analyzer)
{
  for (auto&& arg : args) {
    std::visit([&](auto&& a) { a.analyze(analyzer); }, arg);
  }
}

void pack_args(BufferBuilder& buffer, StoreAnalyzer& analyzer, Span<const Analyzable> args)
{
  buffer.pack<std::uint32_t>(static_cast<std::uint32_t>(args.size()));
  for (auto&& arg : args) {
    std::visit([&](auto&& a) { a.pack(buffer, analyzer); }, arg);
  }
}

void pack_args(BufferBuilder& buffer, Span<const ScalarArg> args)
{
  buffer.pack<std::uint32_t>(static_cast<std::uint32_t>(args.size()));
  for (auto&& arg : args) {
    arg.pack(buffer);
  }
}

}  // namespace

void TaskLauncher::analyze_arguments_(bool parallel, StoreAnalyzer* analyzer)
{
  analyzer->relax_interference_checks(relax_interference_checks_);

  try {
    analyze(inputs_, *analyzer);
    analyze(outputs_, *analyzer);
    analyze(reductions_, *analyzer);
  } catch (const InterferingStoreError&) {
    report_interfering_stores_();
  }

  for (auto&& future : futures_) {
    analyzer->insert(future);
  }

  if (parallel) {
    // Future Maps are employed for tasks that can produce multiple future
    // concurrently. The parallel flag indicates whether this is such a
    // task or not (Index task with point tasks producing futures). In the
    // other case we do not need a Future map as there would be only one
    // Future.
    for (auto&& future_map : future_maps_) {
      analyzer->insert(future_map);
    }
  }

  // Coalesce region requirements before packing task arguments
  // as the latter requires requirement indices to be finalized
  analyzer->analyze();
}

BufferBuilder TaskLauncher::pack_task_arg_(bool parallel, StoreAnalyzer* analyzer)
{
  BufferBuilder task_arg;

  task_arg.pack(library_);
  pack_args(task_arg, *analyzer, inputs_);
  pack_args(task_arg, *analyzer, outputs_);
  pack_args(task_arg, *analyzer, reductions_);
  pack_args(task_arg, scalars_);
  task_arg.pack<std::size_t>(future_size_);
  task_arg.pack<bool>(can_throw_exception_);
  task_arg.pack<bool>(can_elide_device_ctx_sync_);
  if (parallel) {
    // Tasks capable of producing multiple futures (Index task with point tasks
    // producing futures) concurrently are marked as parallel.
    // This kind of task can have one or more communicator as well as barriers.
    task_arg.pack<bool>(insert_barrier_);
    task_arg.pack<std::uint32_t>(static_cast<std::uint32_t>(communicators_.size()));
  } else {
    // insert_barrier
    task_arg.pack<bool>(false);
    // # communicators
    task_arg.pack<std::uint32_t>(0);
  }

  return task_arg;
}

void TaskLauncher::pack_mapper_arg_(BufferBuilder& buffer)
{
  buffer.pack(streaming_generation());
  machine_.pack(buffer);

  std::optional<Legion::ProjectionID> key_proj_id;
  auto find_key_proj_id = [&key_proj_id](Span<const Analyzable> args) {
    for (auto&& arg : args) {
      key_proj_id = std::visit([](auto&& a) { return a.get_key_proj_id(); }, arg);
      if (key_proj_id) {
        break;
      }
    }
  };

  find_key_proj_id(inputs_);
  if (!key_proj_id) {
    find_key_proj_id(outputs_);
  }
  if (!key_proj_id) {
    find_key_proj_id(reductions_);
  }
  if (!key_proj_id) {
    key_proj_id.emplace(0);
  }
  buffer.pack<std::uint32_t>(Runtime::get_runtime().get_sharding(machine_, *key_proj_id));
  buffer.pack(priority_);
}

void TaskLauncher::import_output_regions_(
  Runtime& runtime, const std::vector<Legion::OutputRequirement>& output_requirements)
{
  for (const auto& req : output_requirements) {
    const auto& region = req.parent;
    auto&& region_mgr  = runtime.find_or_create_region_manager(region.get_index_space());

    region_mgr.import_region(region, static_cast<std::uint32_t>(req.privilege_fields.size()));
  };
}

void TaskLauncher::post_process_unbound_stores_(
  const std::vector<Legion::OutputRequirement>& output_requirements)
{
  SmallVector<const OutputRegionArg*> unbound_stores;

  for (auto&& output : outputs_) {
    std::visit([&](auto&& opt) { opt.record_unbound_stores(unbound_stores); }, output);
  }

  if (unbound_stores.empty()) {
    return;
  }

  auto&& runtime = Runtime::get_runtime();
  auto no_part   = create_no_partition();

  import_output_regions_(runtime, output_requirements);

  for (auto&& arg : unbound_stores) {
    LEGATE_ASSERT(arg->requirement_index() != -1U);
    auto* store = arg->store();
    auto& shape = store->shape();
    auto& req   = output_requirements[arg->requirement_index()];

    // This must be done before importing the region field below, as the field manager expects the
    // index space to be available
    if (shape->unbound()) {
      shape->set_index_space(req.parent.get_index_space());
    }

    auto region_field = runtime.import_region_field(
      store->shape(), req.parent, arg->field_id(), store->type()->size());

    store->set_region_field(std::move(region_field));
  }
}

void TaskLauncher::post_process_unbound_store_(const Legion::Domain& launch_domain,
                                               const OutputRegionArg* arg,
                                               const Legion::OutputRequirement& req,
                                               const Legion::FutureMap& weights,
                                               const mapping::detail::Machine& machine,
                                               const ParallelPolicy& parallel_policy)
{
  auto&& runtime  = Runtime::get_runtime();
  auto&& part_mgr = runtime.partition_manager();
  auto* store     = arg->store();
  auto& shape     = store->shape();
  // This must be done before importing the region field below, as the field manager expects the
  // index space to be available
  if (shape->unbound()) {
    shape->set_index_space(req.parent.get_index_space());
  }

  auto region_field =
    runtime.import_region_field(shape, req.parent, arg->field_id(), store->type()->size());
  store->set_region_field(std::move(region_field));

  // TODO(wonchanl): Need to handle key partitions for multi-dimensional unbound stores
  if (store->dim() > 1) {
    return;
  }

  auto partition = create_weighted(weights, launch_domain);
  store->set_key_partition(machine, parallel_policy, partition);

  const auto& index_partition = req.partition.get_index_partition();
  part_mgr.record_index_partition(req.parent.get_index_space(), *partition, index_partition);
}

void TaskLauncher::post_process_unbound_stores_(
  const Legion::FutureMap& result,
  const Legion::Domain& launch_domain,
  const std::vector<Legion::OutputRequirement>& output_requirements)
{
  SmallVector<const OutputRegionArg*> unbound_stores;

  for (auto&& arg : outputs_) {
    std::visit([&](auto&& opt) { opt.record_unbound_stores(unbound_stores); }, arg);
  }

  if (unbound_stores.empty()) {
    return;
  }

  auto&& runtime = Runtime::get_runtime();

  import_output_regions_(runtime, output_requirements);

  if (unbound_stores.size() == 1) {
    auto* arg       = unbound_stores.front();
    const auto& req = output_requirements[arg->requirement_index()];

    post_process_unbound_store_(launch_domain, arg, req, result, machine_, parallel_policy());
  } else {
    for (auto&& [idx, arg] : legate::detail::enumerate(unbound_stores)) {
      const auto& req = output_requirements[arg->requirement_index()];

      if (arg->store()->dim() == 1) {
        post_process_unbound_store_(
          launch_domain,
          arg,
          req,
          runtime.extract_scalar(parallel_policy(),
                                 result,
                                 static_cast<std::uint32_t>(idx) * sizeof(std::size_t),
                                 sizeof(std::size_t),
                                 launch_domain),
          machine_,
          parallel_policy());
      } else {
        post_process_unbound_store_(launch_domain, arg, req, result, machine_, parallel_policy());
      }
    }
  }
}

void TaskLauncher::report_interfering_stores_() const
{
  LEGATE_ABORT(
    "Task ",
    library_.get().get_task_name(task_id_),
    " has interfering store arguments. This means the task tries to access the same store"
    "via multiplepartitions in mixed modes, which is illegal in Legate. Make sure to make a "
    "copy "
    "of the store so there would be no interference.");
}

std::size_t TaskLauncher::get_future_size_including_exception_() const
{
  return future_size_ +
         (static_cast<std::size_t>(can_throw_exception_) * ReturnedException::max_size());
}

}  // namespace legate::detail
