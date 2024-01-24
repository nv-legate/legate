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

#include "core/comm/comm_cpu.h"

#include "core/comm/coll.h"
#include "core/operation/detail/task_launcher.h"
#include "core/runtime/detail/communicator_manager.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/runtime.h"
#include "core/runtime/runtime.h"
#include "core/utilities/detail/malloc.h"

namespace legate::detail {

void show_progress(const Legion::Task* task, Legion::Context ctx, Legion::Runtime* runtime);

}  // namespace legate::detail

namespace legate::comm::cpu {
using Legion::FutureMap;

class Factory final : public detail::CommunicatorFactory {
 public:
  explicit Factory(const detail::Library* core_library);

  [[nodiscard]] bool needs_barrier() const override { return false; }
  [[nodiscard]] bool is_supported_target(mapping::TaskTarget target) const override;

 protected:
  FutureMap initialize(const mapping::detail::Machine& machine, uint32_t num_tasks) override;
  void finalize(const mapping::detail::Machine& machine,
                uint32_t num_tasks,
                const Legion::FutureMap& communicator) override;

 private:
  const detail::Library* core_library_{};
};

Factory::Factory(const detail::Library* core_library) : core_library_{core_library} {}

bool Factory::is_supported_target(mapping::TaskTarget /*target*/) const { return true; }

Legion::FutureMap Factory::initialize(const mapping::detail::Machine& machine, uint32_t num_tasks)
{
  const Domain launch_domain(Rect<1>(Point<1>(0), Point<1>(static_cast<int64_t>(num_tasks) - 1)));
  auto tag =
    machine.preferred_target == mapping::TaskTarget::OMP ? LEGATE_OMP_VARIANT : LEGATE_CPU_VARIANT;

  // Generate a unique ID
  auto comm_id = Legion::Future::from_value<int32_t>(coll::collInitComm());

  // Find a mapping of all participants
  detail::TaskLauncher init_cpucoll_mapping_launcher{
    core_library_, machine, LEGATE_CORE_INIT_CPUCOLL_MAPPING_TASK_ID, tag};
  init_cpucoll_mapping_launcher.add_future(comm_id);
  auto mapping = init_cpucoll_mapping_launcher.execute(launch_domain);

  // Then create communicators on participating processors
  detail::TaskLauncher init_cpucoll_launcher{
    core_library_, machine, LEGATE_CORE_INIT_CPUCOLL_TASK_ID, tag};
  init_cpucoll_launcher.add_future(comm_id);
  init_cpucoll_launcher.set_concurrent(true);

  auto domain = mapping.get_future_map_domain();
  for (Domain::DomainPointIterator it{domain}; it; ++it) {
    init_cpucoll_launcher.add_future(mapping.get_future(*it));
  }
  return init_cpucoll_launcher.execute(launch_domain);
}

void Factory::finalize(const mapping::detail::Machine& machine,
                       uint32_t num_tasks,
                       const Legion::FutureMap& communicator)
{
  const auto tag =
    machine.preferred_target == mapping::TaskTarget::OMP ? LEGATE_OMP_VARIANT : LEGATE_CPU_VARIANT;
  const Domain launch_domain{Rect<1>(Point<1>(0), Point<1>(static_cast<int64_t>(num_tasks) - 1))};
  detail::TaskLauncher launcher{core_library_, machine, LEGATE_CORE_FINALIZE_CPUCOLL_TASK_ID, tag};
  launcher.set_concurrent(true);
  launcher.add_future_map(communicator);
  launcher.execute(launch_domain);
}

namespace {

int init_cpucoll_mapping(const Legion::Task* task,
                         const std::vector<Legion::PhysicalRegion>& /*regions*/,
                         Legion::Context context,
                         Legion::Runtime* runtime)
{
  legate::detail::show_progress(task, context, runtime);
  // clang-tidy cannot see the MPI_Comm_rank() call below
  int mpi_rank = 0;  // NOLINT(misc-const-correctness)
#if LegateDefined(LEGATE_USE_NETWORK)
  if (coll::backend_network->comm_type == coll::CollCommType::CollMPI) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  }
#endif

  return mpi_rank;
}

coll::CollComm init_cpucoll(const Legion::Task* task,
                            const std::vector<Legion::PhysicalRegion>& /*regions*/,
                            Legion::Context context,
                            Legion::Runtime* runtime)
{
  legate::detail::show_progress(task, context, runtime);

  const auto point     = static_cast<int>(task->index_point[0]);
  const auto num_ranks = static_cast<int>(task->index_domain.get_volume());

  assert(task->futures.size() == static_cast<size_t>(num_ranks + 1));
  const int unique_id = task->futures[0].get_result<int>();

  coll::CollComm comm;

  legate::detail::typed_malloc(&comm, 1);
  if (LegateDefined(LEGATE_USE_NETWORK) &&
      (coll::backend_network->comm_type == coll::CollCommType::CollMPI)) {
    int* mapping_table;

    legate::detail::typed_malloc(&mapping_table, num_ranks);
    for (int i = 0; i < num_ranks; i++) {
      const auto mapping_table_element = task->futures[i + 1].get_result<int>();
      mapping_table[i]                 = mapping_table_element;
    }
    auto ret = coll::collCommCreate(comm, num_ranks, point, unique_id, mapping_table);
    assert(ret == coll::CollSuccess);
    assert(mapping_table[point] == comm->mpi_rank);
    free(mapping_table);
  } else {
    auto ret = coll::collCommCreate(comm, num_ranks, point, unique_id, nullptr);
    assert(ret == coll::CollSuccess);
  }

  return comm;
}

void finalize_cpucoll(const Legion::Task* task,
                      const std::vector<Legion::PhysicalRegion>& /*regions*/,
                      Legion::Context context,
                      Legion::Runtime* runtime)
{
  legate::detail::show_progress(task, context, runtime);

  assert(task->futures.size() == 1);
  auto comm        = task->futures[0].get_result<coll::CollComm>();
  const auto point = static_cast<int>(task->index_point[0]);
  assert(comm->global_rank == point);
  auto ret = coll::collCommDestroy(comm);
  assert(ret == coll::CollSuccess);
  free(comm);
  comm = nullptr;
}

}  // namespace

void register_tasks(const detail::Library* core_library)
{
  const auto runtime       = Legion::Runtime::get_runtime();
  const auto& command_args = Legion::Runtime::get_input_args();
  auto ret                 = coll::collInit(command_args.argc, command_args.argv);
  assert(ret == coll::CollSuccess);
  auto init_cpucoll_mapping_task_id =
    core_library->get_task_id(LEGATE_CORE_INIT_CPUCOLL_MAPPING_TASK_ID);
  const char* init_cpucoll_mapping_task_name = "core::comm::cpu::init_mapping";
  runtime->attach_name(init_cpucoll_mapping_task_id,
                       init_cpucoll_mapping_task_name,
                       false /*mutable*/,
                       true /*local only*/);

  auto init_cpucoll_task_id          = core_library->get_task_id(LEGATE_CORE_INIT_CPUCOLL_TASK_ID);
  const char* init_cpucoll_task_name = "core::comm::cpu::init";
  runtime->attach_name(
    init_cpucoll_task_id, init_cpucoll_task_name, false /*mutable*/, true /*local only*/);

  auto finalize_cpucoll_task_id = core_library->get_task_id(LEGATE_CORE_FINALIZE_CPUCOLL_TASK_ID);
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
  {
    auto registrar = make_registrar(
      init_cpucoll_mapping_task_id, init_cpucoll_mapping_task_name, Processor::TOC_PROC);
    runtime->register_task_variant<int, init_cpucoll_mapping>(registrar, LEGATE_GPU_VARIANT);
  }
  {
    auto registrar =
      make_registrar(init_cpucoll_task_id, init_cpucoll_task_name, Processor::TOC_PROC);
    runtime->register_task_variant<coll::CollComm, init_cpucoll>(registrar, LEGATE_GPU_VARIANT);
  }
  {
    auto registrar =
      make_registrar(finalize_cpucoll_task_id, finalize_cpucoll_task_name, Processor::TOC_PROC);
    runtime->register_task_variant<finalize_cpucoll>(registrar, LEGATE_GPU_VARIANT);
  }
}

void register_factory(const detail::Library* library)
{
  auto* comm_mgr = detail::Runtime::get_runtime()->communicator_manager();
  comm_mgr->register_factory("cpu", std::make_unique<Factory>(library));
}

}  // namespace legate::comm::cpu
