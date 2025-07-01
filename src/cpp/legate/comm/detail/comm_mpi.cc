/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/comm_mpi.h>

#include <legate_defines.h>

#include <legate/comm/coll.h>
#include <legate/comm/coll_comm.h>
#include <legate/comm/detail/backend_network.h>
#include <legate/comm/detail/comm_cpu_factory.h>
#include <legate/comm/detail/mpi_interface.h>
#include <legate/task/detail/legion_task.h>
#include <legate/task/task_config.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/macros.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>

namespace legate::detail {

void show_progress(const Legion::Task* task, Legion::Context ctx, Legion::Runtime* runtime);

}  // namespace legate::detail

namespace legate::detail::comm::mpi {

class InitMapping : public legate::detail::LegionTask<InitMapping> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{LocalTaskID{CoreTask::INIT_CPUCOLL_MAPPING}}.with_variant_options(
      legate::VariantOptions{}.with_elide_device_ctx_sync(true));

  static int cpu_variant(const Legion::Task* task,
                         const std::vector<Legion::PhysicalRegion>& /*regions*/,
                         Legion::Context context,
                         Legion::Runtime* runtime)
  {
    legate::detail::show_progress(task, context, runtime);

    LEGATE_CHECK(coll::BackendNetwork::get_network()->comm_type ==
                 legate::comm::coll::CollCommType::CollMPI);

    int mpi_rank;

    LEGATE_CHECK_MPI(
      detail::MPIInterface::mpi_comm_rank(detail::MPIInterface::MPI_COMM_WORLD(), &mpi_rank));
    return mpi_rank;
  }

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static int omp_variant(const Legion::Task* task,
                         const std::vector<Legion::PhysicalRegion>& regions,
                         Legion::Context context,
                         Legion::Runtime* runtime)
  {
    return cpu_variant(task, regions, context, runtime);
  }
#endif

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static int gpu_variant(const Legion::Task* task,
                         const std::vector<Legion::PhysicalRegion>& regions,
                         Legion::Context context,
                         Legion::Runtime* runtime)
  {
    return cpu_variant(task, regions, context, runtime);
  }
#endif
};

class Init : public legate::detail::LegionTask<Init> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{LocalTaskID{CoreTask::INIT_CPUCOLL}}.with_variant_options(
      legate::VariantOptions{}.with_concurrent(true));

  static legate::comm::coll::CollComm cpu_variant(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& /*regions*/,
    Legion::Context context,
    Legion::Runtime* runtime)
  {
    legate::detail::show_progress(task, context, runtime);

    LEGATE_CHECK(coll::BackendNetwork::get_network()->comm_type ==
                 legate::comm::coll::CollCommType::CollMPI);

    const auto point     = static_cast<std::size_t>(task->index_point[0]);
    const auto num_ranks = task->index_domain.get_volume();

    LEGATE_CHECK(task->futures.size() == num_ranks + 1);

    const auto unique_id     = task->futures[0].get_result<int>();
    auto comm                = std::make_unique<legate::comm::coll::Coll_Comm>();
    const auto mapping_table = make_mapping_table_(num_ranks, *task);

    legate::comm::coll::collCommCreate(comm.get(),
                                       static_cast<int>(num_ranks),
                                       static_cast<int>(point),
                                       unique_id,
                                       mapping_table.data());
    return comm.release();
  }

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static legate::comm::coll::CollComm omp_variant(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context context,
    Legion::Runtime* runtime)
  {
    return cpu_variant(task, regions, context, runtime);
  }
#endif

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static legate::comm::coll::CollComm gpu_variant(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context context,
    Legion::Runtime* runtime)
  {
    return cpu_variant(task, regions, context, runtime);
  }
#endif

 private:
  [[nodiscard]] static std::vector<int> make_mapping_table_(std::size_t size,
                                                            const Legion::Task& task)
  {
    std::vector<int> ret;

    ret.reserve(size);
    std::generate_n(std::back_inserter(ret), size, [&, i = std::size_t{1}]() mutable {
      return task.futures[i++].get_result<int>();
    });
    return ret;
  }
};

class Finalize : public legate::detail::LegionTask<Finalize> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{LocalTaskID{CoreTask::FINALIZE_CPUCOLL}}.with_variant_options(
      legate::VariantOptions{}.with_concurrent(true));

  static void cpu_variant(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& /*regions*/,
                          Legion::Context context,
                          Legion::Runtime* runtime)
  {
    legate::detail::show_progress(task, context, runtime);

    LEGATE_CHECK(task->futures.size() == 1);
    std::unique_ptr<legate::comm::coll::Coll_Comm> comm{
      task->futures[0].get_result<legate::comm::coll::CollComm>()};

    LEGATE_CHECK(comm->global_rank == static_cast<int>(task->index_point[0]));
    legate::comm::coll::collCommDestroy(comm.get());
  }

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& regions,
                          Legion::Context context,
                          Legion::Runtime* runtime)
  {
    cpu_variant(task, regions, context, runtime);
  }
#endif

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& regions,
                          Legion::Context context,
                          Legion::Runtime* runtime)
  {
    cpu_variant(task, regions, context, runtime);
  }
#endif
};

void register_tasks(const legate::Library& core_library)
{
  InitMapping::register_variants(core_library);
  Init::register_variants(core_library);
  Finalize::register_variants(core_library);
}

std::unique_ptr<CommunicatorFactory> make_factory(const legate::detail::Library& library)
{
  return std::make_unique<cpu::Factory<Init, InitMapping, Finalize>>(library);
}

}  // namespace legate::detail::comm::mpi
