/* Copyright 2022 NVIDIA Corporation
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

#include "core/comm/comm_cpu.h"
#include "core/operation/detail/task_launcher.h"
#include "core/runtime/detail/communicator_manager.h"
#include "core/runtime/detail/runtime.h"
#include "legate.h"

#include "core/comm/coll.h"

namespace legate::comm::cpu {

class Factory : public detail::CommunicatorFactory {
 public:
  Factory(const LibraryContext* core_context);

 public:
  bool needs_barrier() const override { return false; }
  bool is_supported_target(mapping::TaskTarget target) const override;

 protected:
  Legion::FutureMap initialize(const mapping::MachineDesc& machine, uint32_t num_tasks) override;
  void finalize(const mapping::MachineDesc& machine,
                uint32_t num_tasks,
                const Legion::FutureMap& communicator) override;

 private:
  const LibraryContext* core_context_;
};

Factory::Factory(const LibraryContext* core_context) : core_context_(core_context) {}

bool Factory::is_supported_target(mapping::TaskTarget target) const
{
  return target == mapping::TaskTarget::OMP || target == mapping::TaskTarget::CPU;
}

Legion::FutureMap Factory::initialize(const mapping::MachineDesc& machine, uint32_t num_tasks)
{
  Domain launch_domain(Rect<1>(Point<1>(0), Point<1>(static_cast<int64_t>(num_tasks) - 1)));
  auto tag =
    machine.preferred_target == mapping::TaskTarget::OMP ? LEGATE_OMP_VARIANT : LEGATE_CPU_VARIANT;

  // Generate a unique ID
  auto comm_id = Legion::Future::from_value<int32_t>(coll::collInitComm());

  // Find a mapping of all participants
  detail::TaskLauncher init_cpucoll_mapping_launcher(
    core_context_, machine, LEGATE_CORE_INIT_CPUCOLL_MAPPING_TASK_ID, tag);
  init_cpucoll_mapping_launcher.add_future(comm_id);
  auto mapping = init_cpucoll_mapping_launcher.execute(launch_domain);

  // Then create communicators on participating processors
  detail::TaskLauncher init_cpucoll_launcher(
    core_context_, machine, LEGATE_CORE_INIT_CPUCOLL_TASK_ID, tag);
  init_cpucoll_launcher.add_future(comm_id);
  init_cpucoll_launcher.set_concurrent(true);

  auto domain = mapping.get_future_map_domain();
  for (Domain::DomainPointIterator it(domain); it; ++it)
    init_cpucoll_launcher.add_future(mapping.get_future(*it));
  return init_cpucoll_launcher.execute(launch_domain);
}

void Factory::finalize(const mapping::MachineDesc& machine,
                       uint32_t num_tasks,
                       const Legion::FutureMap& communicator)
{
  auto tag =
    machine.preferred_target == mapping::TaskTarget::OMP ? LEGATE_OMP_VARIANT : LEGATE_CPU_VARIANT;
  Domain launch_domain(Rect<1>(Point<1>(0), Point<1>(static_cast<int64_t>(num_tasks) - 1)));
  detail::TaskLauncher launcher(core_context_, machine, LEGATE_CORE_FINALIZE_CPUCOLL_TASK_ID, tag);
  launcher.set_concurrent(true);
  launcher.add_future_map(communicator);
  launcher.execute(launch_domain);
}

static int init_cpucoll_mapping(const Legion::Task* task,
                                const std::vector<Legion::PhysicalRegion>& regions,
                                Legion::Context context,
                                Legion::Runtime* runtime)
{
  Core::show_progress(task, context, runtime);
  int mpi_rank = 0;
#if defined(LEGATE_USE_NETWORK)
  if (coll::backend_network->comm_type == coll::CollCommType::CollMPI) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  }
#endif

  return mpi_rank;
}

static coll::CollComm init_cpucoll(const Legion::Task* task,
                                   const std::vector<Legion::PhysicalRegion>& regions,
                                   Legion::Context context,
                                   Legion::Runtime* runtime)
{
  Core::show_progress(task, context, runtime);

  const int point = task->index_point[0];
  int num_ranks   = task->index_domain.get_volume();

  assert(task->futures.size() == static_cast<size_t>(num_ranks + 1));
  const int unique_id = task->futures[0].get_result<int>();

  coll::CollComm comm = (coll::CollComm)malloc(sizeof(coll::Coll_Comm));

#ifdef LEGATE_USE_NETWORK
  if (coll::backend_network->comm_type == coll::CollCommType::CollMPI) {
    int* mapping_table = (int*)malloc(sizeof(int) * num_ranks);
    for (int i = 0; i < num_ranks; i++) {
      const int mapping_table_element = task->futures[i + 1].get_result<int>();
      mapping_table[i]                = mapping_table_element;
    }
    coll::collCommCreate(comm, num_ranks, point, unique_id, mapping_table);
    assert(mapping_table[point] == comm->mpi_rank);
    free(mapping_table);
  } else
#endif
  {
    coll::collCommCreate(comm, num_ranks, point, unique_id, nullptr);
  }

  return comm;
}

static void finalize_cpucoll(const Legion::Task* task,
                             const std::vector<Legion::PhysicalRegion>& regions,
                             Legion::Context context,
                             Legion::Runtime* runtime)
{
  Core::show_progress(task, context, runtime);

  assert(task->futures.size() == 1);
  coll::CollComm comm = task->futures[0].get_result<coll::CollComm>();
  const int point     = task->index_point[0];
  assert(comm->global_rank == point);
  coll::collCommDestroy(comm);
  free(comm);
  comm = nullptr;
}

void register_tasks(Legion::Machine machine,
                    Legion::Runtime* runtime,
                    const LibraryContext* context)
{
  const auto& command_args = Legion::Runtime::get_input_args();
  coll::collInit(command_args.argc, command_args.argv);

  auto init_cpucoll_mapping_task_id =
    context->get_task_id(LEGATE_CORE_INIT_CPUCOLL_MAPPING_TASK_ID);
  const char* init_cpucoll_mapping_task_name = "core::comm::cpu::init_mapping";
  runtime->attach_name(init_cpucoll_mapping_task_id,
                       init_cpucoll_mapping_task_name,
                       false /*mutable*/,
                       true /*local only*/);

  auto init_cpucoll_task_id          = context->get_task_id(LEGATE_CORE_INIT_CPUCOLL_TASK_ID);
  const char* init_cpucoll_task_name = "core::comm::cpu::init";
  runtime->attach_name(
    init_cpucoll_task_id, init_cpucoll_task_name, false /*mutable*/, true /*local only*/);

  auto finalize_cpucoll_task_id = context->get_task_id(LEGATE_CORE_FINALIZE_CPUCOLL_TASK_ID);
  const char* finalize_cpucoll_task_name = "core::comm::cpu::finalize";
  runtime->attach_name(
    finalize_cpucoll_task_id, finalize_cpucoll_task_name, false /*mutable*/, true /*local only*/);

  auto make_registrar = [&](auto task_id, auto* task_name, auto proc_kind) {
    Legion::TaskVariantRegistrar registrar(task_id, task_name);
    registrar.add_constraint(Legion::ProcessorConstraint(proc_kind));
    registrar.set_leaf(true);
    registrar.global_registration = false;
    return registrar;
  };

  // Register the task variants
  {
    auto registrar = make_registrar(
      init_cpucoll_mapping_task_id, init_cpucoll_mapping_task_name, Processor::LOC_PROC);
    runtime->register_task_variant<int, init_cpucoll_mapping>(registrar, LEGATE_CPU_VARIANT);
  }
  {
    auto registrar =
      make_registrar(init_cpucoll_task_id, init_cpucoll_task_name, Processor::LOC_PROC);
    runtime->register_task_variant<coll::CollComm, init_cpucoll>(registrar, LEGATE_CPU_VARIANT);
  }
  {
    auto registrar =
      make_registrar(finalize_cpucoll_task_id, finalize_cpucoll_task_name, Processor::LOC_PROC);
    runtime->register_task_variant<finalize_cpucoll>(registrar, LEGATE_CPU_VARIANT);
  }
  {
    auto registrar = make_registrar(
      init_cpucoll_mapping_task_id, init_cpucoll_mapping_task_name, Processor::OMP_PROC);
    runtime->register_task_variant<int, init_cpucoll_mapping>(registrar, LEGATE_OMP_VARIANT);
  }
  {
    auto registrar =
      make_registrar(init_cpucoll_task_id, init_cpucoll_task_name, Processor::OMP_PROC);
    runtime->register_task_variant<coll::CollComm, init_cpucoll>(registrar, LEGATE_OMP_VARIANT);
  }
  {
    auto registrar =
      make_registrar(finalize_cpucoll_task_id, finalize_cpucoll_task_name, Processor::OMP_PROC);
    runtime->register_task_variant<finalize_cpucoll>(registrar, LEGATE_OMP_VARIANT);
  }
}

void register_factory(const LibraryContext* context)
{
  auto* comm_mgr = detail::Runtime::get_runtime()->communicator_manager();
  comm_mgr->register_factory("cpu", std::make_unique<Factory>(context));
}

}  // namespace legate::comm::cpu
