/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "core/runtime/detail/launcher.h"
#include "core/data/detail/logical_region_field.h"
#include "core/data/detail/logical_store.h"
#include "core/data/scalar.h"
#include "core/mapping/machine.h"
#include "core/runtime/context.h"
#include "core/runtime/detail/launcher_arg.h"
#include "core/runtime/detail/partition_manager.h"
#include "core/runtime/detail/req_analyzer.h"
#include "core/runtime/detail/runtime.h"
#include "core/utilities/buffer_builder.h"

namespace legate::detail {

TaskLauncher::TaskLauncher(const LibraryContext* library,
                           const mapping::MachineDesc& machine,
                           int64_t task_id,
                           int64_t tag /*= 0*/)
  : library_(library), task_id_(task_id), tag_(tag), machine_(machine), provenance_("")
{
  initialize();
}

TaskLauncher::TaskLauncher(const LibraryContext* library,
                           const mapping::MachineDesc& machine,
                           const std::string& provenance,
                           int64_t task_id,
                           int64_t tag /*= 0*/)
  : library_(library), task_id_(task_id), tag_(tag), machine_(machine), provenance_(provenance)
{
  initialize();
}

TaskLauncher::~TaskLauncher()
{
  for (auto& arg : inputs_) delete arg;
  for (auto& arg : outputs_) delete arg;
  for (auto& arg : reductions_) delete arg;
  for (auto& arg : scalars_) delete arg;

  delete req_analyzer_;
  delete buffer_;
  delete mapper_arg_;
}

void TaskLauncher::initialize()
{
  req_analyzer_ = new RequirementAnalyzer();
  out_analyzer_ = new OutputRequirementAnalyzer();
  buffer_       = new BufferBuilder();
  mapper_arg_   = new BufferBuilder();
  machine_.pack(*mapper_arg_);
}

int64_t TaskLauncher::legion_task_id() const { return library_->get_task_id(task_id_); }

int64_t TaskLauncher::legion_mapper_id() const { return library_->get_mapper_id(); }

void TaskLauncher::add_scalar(const Scalar& scalar)
{
  scalars_.push_back(new UntypedScalarArg(scalar));
}

void TaskLauncher::add_input(LogicalStore* store,
                             std::unique_ptr<Projection> proj,
                             Legion::MappingTagID tag,
                             Legion::RegionFlags flags)
{
  add_store(inputs_, store, std::move(proj), READ_ONLY, tag, flags);
}

void TaskLauncher::add_output(LogicalStore* store,
                              std::unique_ptr<Projection> proj,
                              Legion::MappingTagID tag,
                              Legion::RegionFlags flags)
{
  add_store(outputs_, store, std::move(proj), WRITE_ONLY, tag, flags);
}

void TaskLauncher::add_reduction(LogicalStore* store,
                                 std::unique_ptr<Projection> proj,
                                 bool read_write,
                                 Legion::MappingTagID tag,
                                 Legion::RegionFlags flags)
{
  if (read_write)
    add_store(reductions_, store, std::move(proj), READ_WRITE, tag, flags);
  else
    add_store(reductions_, store, std::move(proj), REDUCE, tag, flags);
}

void TaskLauncher::add_unbound_output(LogicalStore* store,
                                      Legion::FieldSpace field_space,
                                      Legion::FieldID field_id)
{
  out_analyzer_->insert(store->dim(), field_space, field_id);
  auto arg = new OutputRegionArg(out_analyzer_, store, field_space, field_id);
  outputs_.push_back(arg);
  unbound_stores_.push_back(arg);
}

void TaskLauncher::add_future(const Legion::Future& future)
{
  // FIXME: Futures that are directly added by this function are incompatible with those
  // from scalar stores. We need to separate the two sets.
  futures_.push_back(future);
}

void TaskLauncher::add_future_map(const Legion::FutureMap& future_map)
{
  future_maps_.push_back(future_map);
}

void TaskLauncher::add_communicator(const Legion::FutureMap& communicator)
{
  communicators_.push_back(communicator);
}

Legion::FutureMap TaskLauncher::execute(const Legion::Domain& launch_domain)
{
  auto legion_launcher = build_index_task(launch_domain);

  if (output_requirements_.empty()) return Runtime::get_runtime()->dispatch(legion_launcher.get());

  auto result = Runtime::get_runtime()->dispatch(legion_launcher.get(), &output_requirements_);
  post_process_unbound_stores(result, launch_domain);
  return result;
}

Legion::Future TaskLauncher::execute_single()
{
  auto legion_launcher = build_single_task();

  if (output_requirements_.empty()) return Runtime::get_runtime()->dispatch(legion_launcher.get());
  auto result = Runtime::get_runtime()->dispatch(legion_launcher.get(), &output_requirements_);
  post_process_unbound_stores();
  return result;
}

void TaskLauncher::add_store(std::vector<ArgWrapper*>& args,
                             LogicalStore* store,
                             std::unique_ptr<Projection> proj,
                             Legion::PrivilegeMode privilege,
                             Legion::MappingTagID tag,
                             Legion::RegionFlags flags)
{
  auto redop = proj->redop;

  if (store->has_scalar_storage()) {
    auto has_storage = privilege != WRITE_ONLY;
    auto read_only   = privilege == READ_ONLY;
    if (has_storage) futures_.push_back(store->get_future());
    args.push_back(new FutureStoreArg(store, read_only, has_storage, redop));
  } else {
    auto region_field = store->get_region_field();
    auto region       = region_field->region();
    auto field_id     = region_field->field_id();

    auto proj_info = new ProjectionInfo(proj.get(), tag, flags);

    req_analyzer_->insert(region, field_id, privilege, proj_info);
    args.push_back(new RegionFieldArg(req_analyzer_, store, field_id, privilege, proj_info));
  }
}

void TaskLauncher::pack_args(const std::vector<ArgWrapper*>& args)
{
  buffer_->pack<uint32_t>(args.size());
  for (auto& arg : args) arg->pack(*buffer_);
}

void TaskLauncher::pack_sharding_functor_id()
{
  // TODO: Generate the right sharding functor id
  mapper_arg_->pack<uint32_t>(0);
}

std::unique_ptr<Legion::TaskLauncher> TaskLauncher::build_single_task()
{
  // Coalesce region requirements before packing task arguments
  // as the latter requires requirement indices to be finalized
  req_analyzer_->analyze_requirements();
  out_analyzer_->analyze_requirements();

  pack_args(inputs_);
  pack_args(outputs_);
  pack_args(reductions_);
  pack_args(scalars_);
  buffer_->pack<bool>(can_throw_exception_);
  // insert_barrier
  buffer_->pack<bool>(false);
  // # communicators
  buffer_->pack<uint32_t>(0);

  pack_sharding_functor_id();
  auto* runtime = Runtime::get_runtime();

  auto single_task = std::make_unique<Legion::TaskLauncher>(legion_task_id(),
                                                            buffer_->to_legion_buffer(),
                                                            Legion::Predicate::TRUE_PRED,
                                                            legion_mapper_id(),
                                                            tag_,
                                                            mapper_arg_->to_legion_buffer(),
                                                            provenance_.c_str());
  for (auto& future : futures_) single_task->add_future(future);

  req_analyzer_->populate_launcher(single_task.get());
  out_analyzer_->populate_output_requirements(output_requirements_);

  single_task->local_function_task =
    !has_side_effect_ && req_analyzer_->empty() && out_analyzer_->empty();

  return single_task;
}

std::unique_ptr<Legion::IndexTaskLauncher> TaskLauncher::build_index_task(
  const Legion::Domain& launch_domain)
{
  // Coalesce region requirements before packing task arguments
  // as the latter requires requirement indices to be finalized
  req_analyzer_->analyze_requirements();
  out_analyzer_->analyze_requirements();

  pack_args(inputs_);
  pack_args(outputs_);
  pack_args(reductions_);
  pack_args(scalars_);
  buffer_->pack<bool>(can_throw_exception_);
  buffer_->pack<bool>(insert_barrier_);
  buffer_->pack<uint32_t>(communicators_.size());

  pack_sharding_functor_id();
  auto* runtime = Runtime::get_runtime();

  auto index_task = std::make_unique<Legion::IndexTaskLauncher>(legion_task_id(),
                                                                launch_domain,
                                                                buffer_->to_legion_buffer(),
                                                                Legion::ArgumentMap(),
                                                                Legion::Predicate::TRUE_PRED,
                                                                false /*must*/,
                                                                legion_mapper_id(),
                                                                tag_,
                                                                mapper_arg_->to_legion_buffer(),
                                                                provenance_.c_str());
  for (auto& future : futures_) index_task->add_future(future);
  if (insert_barrier_) {
    size_t num_tasks                     = launch_domain.get_volume();
    auto [arrival_barrier, wait_barrier] = runtime->create_barriers(num_tasks);
    index_task->add_future(Legion::Future::from_value(arrival_barrier));
    index_task->add_future(Legion::Future::from_value(wait_barrier));
    runtime->destroy_barrier(arrival_barrier);
    runtime->destroy_barrier(wait_barrier);
  }
  for (auto& communicator : communicators_) index_task->point_futures.push_back(communicator);
  for (auto& future_map : future_maps_) index_task->point_futures.push_back(future_map);

  req_analyzer_->populate_launcher(index_task.get());
  out_analyzer_->populate_output_requirements(output_requirements_);

  index_task->concurrent = concurrent_ || !communicators_.empty();

  return index_task;
}

void TaskLauncher::bind_region_fields_to_unbound_stores() {}

void TaskLauncher::post_process_unbound_stores()
{
  if (unbound_stores_.empty()) return;

  auto* runtime = Runtime::get_runtime();
  auto no_part  = create_no_partition();

  for (auto& arg : unbound_stores_) {
#ifdef DEBUG_LEGATE
    assert(arg->requirement_index() != -1U);
#endif
    auto* store = arg->store();
    auto& req   = output_requirements_[arg->requirement_index()];
    auto region_field =
      runtime->import_region_field(req.parent, arg->field_id(), store->type().size());
    store->set_region_field(std::move(region_field));
    store->set_key_partition(machine_, no_part.get());
  }
}

void TaskLauncher::post_process_unbound_stores(const Legion::FutureMap& result,
                                               const Legion::Domain& launch_domain)
{
  if (unbound_stores_.empty()) return;

  auto* runtime  = Runtime::get_runtime();
  auto* part_mgr = runtime->partition_manager();

  auto post_process_unbound_store =
    [&runtime, &part_mgr, &launch_domain](
      auto*& arg, const auto& req, const auto& weights, const auto& machine) {
      auto* store = arg->store();
      auto region_field =
        runtime->import_region_field(req.parent, arg->field_id(), store->type().size());
      store->set_region_field(std::move(region_field));

      // TODO: Need to handle key partitions for multi-dimensional unbound stores
      if (store->dim() > 1) return;

      auto partition = create_weighted(weights, launch_domain);
      store->set_key_partition(machine, partition.get());

      const auto& index_partition = req.partition.get_index_partition();
      part_mgr->record_index_partition(req.parent.get_index_space(), *partition, index_partition);
    };

  if (unbound_stores_.size() == 1) {
    auto* arg       = unbound_stores_.front();
    const auto& req = output_requirements_[arg->requirement_index()];
    post_process_unbound_store(arg, req, result, machine_);
  } else {
    uint32_t idx = 0;
    for (auto& arg : unbound_stores_) {
      const auto& req = output_requirements_[arg->requirement_index()];
      if (arg->store()->dim() == 1)
        post_process_unbound_store(
          arg, req, runtime->extract_scalar(result, idx, launch_domain), machine_);
      else
        post_process_unbound_store(arg, req, result, machine_);
      ++idx;
    }
  }
}

}  // namespace legate::detail
