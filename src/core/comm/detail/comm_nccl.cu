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

#include "core/comm/detail/comm_nccl.h"
#include "core/cuda/cuda.h"
#include "core/data/buffer.h"
#include "core/operation/detail/task_launcher.h"
#include "core/runtime/detail/communicator_manager.h"
#include "core/runtime/detail/config.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/runtime.h"
#include "core/runtime/runtime.h"
#include "core/task/detail/legion_task.h"
#include "core/utilities/nvtx_help.h"
#include "core/utilities/typedefs.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <memory>
#include <nccl.h>
#include <string>
#include <vector>

namespace legate::detail {

void show_progress(const Legion::Task* task, Legion::Context ctx, Legion::Runtime* runtime);

namespace comm::nccl {

namespace {

#define LEGATE_CHECK_NCCL(...)               \
  do {                                       \
    const ncclResult_t result = __VA_ARGS__; \
    check_nccl(result, __FILE__, __LINE__);  \
  } while (false)

void check_nccl(ncclResult_t error, const char* file, int line)
{
  if (error != ncclSuccess) {
    static_cast<void>(fprintf(stderr,
                              "Internal NCCL failure with error %d (%s) in file %s at line %d\n",
                              error,
                              ncclGetErrorString(error),
                              file,
                              line));
    std::exit(error);
  }
}

class Payload {
 public:
  std::uint64_t field0;
  std::uint64_t field1;
};

}  // namespace

class InitId : public detail::LegionTask<InitId> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{LEGATE_CORE_INIT_NCCL_ID_TASK_ID};

  static ncclUniqueId gpu_variant(const Legion::Task* task,
                                  const std::vector<Legion::PhysicalRegion>& /*regions*/,
                                  Legion::Context context,
                                  Legion::Runtime* runtime)
  {
    const legate::nvtx::Range auto_range{task_name_().data()};

    legate::detail::show_progress(task, context, runtime);

    ncclUniqueId id;
    LEGATE_CHECK_NCCL(ncclGetUniqueId(&id));

    return id;
  }
};

class Init : public detail::LegionTask<Init> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{LEGATE_CORE_INIT_NCCL_TASK_ID};

  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);

  static ncclComm_t* gpu_variant(const Legion::Task* task,
                                 const std::vector<Legion::PhysicalRegion>& /*regions*/,
                                 Legion::Context context,
                                 Legion::Runtime* runtime)
  {
    const legate::nvtx::Range auto_range{task_name_().data()};

    legate::detail::show_progress(task, context, runtime);

    LEGATE_CHECK(task->futures.size() == 1);

    auto id   = task->futures[0].get_result<ncclUniqueId>();
    auto comm = std::make_unique<ncclComm_t>();

    auto num_ranks = task->index_domain.get_volume();
    auto rank_id   = task->index_point[0];

    auto ts_init_start = std::chrono::high_resolution_clock::now();
    LEGATE_CHECK_NCCL(ncclGroupStart());
    LEGATE_CHECK_NCCL(ncclCommInitRank(comm.get(), num_ranks, id, rank_id));
    LEGATE_CHECK_NCCL(ncclGroupEnd());
    auto ts_init_stop = std::chrono::high_resolution_clock::now();

    auto time_init = std::chrono::duration<double>(ts_init_stop - ts_init_start).count() * 1000.0;

    if (0 == rank_id) {
      legate::detail::log_legate().debug("NCCL initialization took %lf ms", time_init);
    }

    if (num_ranks == 1) {
      return comm.release();
    }

    if (!detail::Config::warmup_nccl) {
      return comm.release();
    }

    auto stream = detail::Runtime::get_runtime()->get_cuda_stream();

    // Perform a warm-up all-to-all

    cudaEvent_t ev_start, ev_end_all_to_all, ev_end_all_gather;
    LEGATE_CHECK_CUDA(cudaEventCreate(&ev_start));
    LEGATE_CHECK_CUDA(cudaEventCreate(&ev_end_all_to_all));
    LEGATE_CHECK_CUDA(cudaEventCreate(&ev_end_all_gather));

    auto src_buffer = create_buffer<Payload>(num_ranks, Memory::Kind::GPU_FB_MEM);
    auto tgt_buffer = create_buffer<Payload>(num_ranks, Memory::Kind::GPU_FB_MEM);

    LEGATE_CHECK_CUDA(cudaEventRecord(ev_start, stream));

    LEGATE_CHECK_NCCL(ncclGroupStart());
    for (std::size_t idx = 0; idx < num_ranks; ++idx) {
      LEGATE_CHECK_NCCL(ncclSend(src_buffer.ptr(0), sizeof(Payload), ncclInt8, idx, *comm, stream));
      LEGATE_CHECK_NCCL(ncclRecv(tgt_buffer.ptr(0), sizeof(Payload), ncclInt8, idx, *comm, stream));
    }
    LEGATE_CHECK_NCCL(ncclGroupEnd());

    LEGATE_CHECK_CUDA(cudaEventRecord(ev_end_all_to_all, stream));

    LEGATE_CHECK_NCCL(
      ncclAllGather(src_buffer.ptr(0), tgt_buffer.ptr(0), 1, ncclUint64, *comm, stream));

    LEGATE_CHECK_CUDA(cudaEventRecord(ev_end_all_gather, stream));

    LEGATE_CHECK_CUDA(cudaEventSynchronize(ev_end_all_gather));

    float time_all_to_all = 0.;
    float time_all_gather = 0.;
    LEGATE_CHECK_CUDA(cudaEventElapsedTime(&time_all_to_all, ev_start, ev_end_all_to_all));
    LEGATE_CHECK_CUDA(cudaEventElapsedTime(&time_all_gather, ev_end_all_to_all, ev_end_all_gather));

    if (0 == rank_id) {
      legate::detail::log_legate().debug(
        "NCCL warm-up took %f ms (all-to-all: %f ms, all-gather: %f ms)",
        time_all_to_all + time_all_gather,
        time_all_to_all,
        time_all_gather);
    }

    LEGATE_CHECK_CUDA(cudaEventDestroy(ev_start));
    LEGATE_CHECK_CUDA(cudaEventDestroy(ev_end_all_to_all));
    LEGATE_CHECK_CUDA(cudaEventDestroy(ev_end_all_gather));
    return comm.release();
  }
};

class Finalize : public detail::LegionTask<Finalize> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{LEGATE_CORE_FINALIZE_NCCL_TASK_ID};

  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);

  static void gpu_variant(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& /*regions*/,
                          Legion::Context context,
                          Legion::Runtime* runtime)
  {
    const legate::nvtx::Range auto_range{task_name_().data()};

    legate::detail::show_progress(task, context, runtime);

    LEGATE_CHECK(task->futures.size() == 1);
    auto comm = task->futures[0].get_result<ncclComm_t*>();
    LEGATE_CHECK_NCCL(ncclCommDestroy(*comm));
    delete comm;
  }
};

class Factory final : public detail::CommunicatorFactory {
 public:
  explicit Factory(const detail::Library* core_library);

  [[nodiscard]] bool needs_barrier() const override;
  [[nodiscard]] bool is_supported_target(mapping::TaskTarget target) const override;

 private:
  [[nodiscard]] Legion::FutureMap initialize_(const mapping::detail::Machine& machine,
                                              std::uint32_t num_tasks) override;
  void finalize_(const mapping::detail::Machine& machine,
                 std::uint32_t num_tasks,
                 const Legion::FutureMap& communicator) override;

  const detail::Library* core_library_{};
};

Factory::Factory(const detail::Library* core_library) : core_library_{core_library} {}

bool Factory::needs_barrier() const
{
  // Blocking communications in NCCL violate CUDA's (undocumented) concurrent forward progress
  // requirements and no CUDA drivers that have released are safe from this. Until either CUDA
  // or NCCL is fixed, we will always insert a barrier at the beginning of every NCCL task.
  return true;
}

bool Factory::is_supported_target(mapping::TaskTarget target) const
{
  return target == mapping::TaskTarget::GPU;
}

Legion::FutureMap Factory::initialize_(const mapping::detail::Machine& machine,
                                       std::uint32_t num_tasks)
{
  Domain launch_domain{Rect<1>{Point<1>{0}, Point<1>{static_cast<std::int64_t>(num_tasks) - 1}}};

  // Create a communicator ID
  detail::TaskLauncher init_nccl_id_launcher{
    core_library_, machine, InitId::TASK_ID, LEGATE_GPU_VARIANT};
  init_nccl_id_launcher.set_side_effect(true);
  auto nccl_id = init_nccl_id_launcher.execute_single();

  // Then create the communicators on participating GPUs
  detail::TaskLauncher init_nccl_launcher{
    core_library_, machine, Init::TASK_ID, LEGATE_GPU_VARIANT};
  init_nccl_launcher.add_future(nccl_id);
  init_nccl_launcher.set_concurrent(true);
  return init_nccl_launcher.execute(launch_domain);
}

void Factory::finalize_(const mapping::detail::Machine& machine,
                        std::uint32_t num_tasks,
                        const Legion::FutureMap& communicator)
{
  Domain launch_domain{Rect<1>{Point<1>{0}, Point<1>{static_cast<std::int64_t>(num_tasks) - 1}}};

  detail::TaskLauncher launcher{core_library_, machine, Finalize::TASK_ID, LEGATE_GPU_VARIANT};
  launcher.set_concurrent(true);
  launcher.add_future_map(communicator);
  launcher.execute(launch_domain);
}

namespace {

}  // namespace

void register_tasks(detail::Library* core_library)
{
  // Register the task variants
  InitId::register_variants(legate::Library{core_library});
  Init::register_variants(legate::Library{core_library});
  Finalize::register_variants(legate::Library{core_library});
}

void register_factory(const detail::Library* core_library)
{
  auto* comm_mgr = detail::Runtime::get_runtime()->communicator_manager();
  comm_mgr->register_factory("nccl", std::make_unique<Factory>(core_library));
}

}  // namespace comm::nccl

}  // namespace legate::detail
