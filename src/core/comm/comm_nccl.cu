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

#include "core/comm/comm_nccl.h"
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#include "core/data/buffer.h"
#include "core/operation/detail/task_launcher.h"
#include "core/runtime/detail/communicator_manager.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/runtime.h"
#include "core/runtime/runtime.h"
#include "core/utilities/nvtx_help.h"
#include "core/utilities/typedefs.h"

#include <chrono>
#include <cuda.h>
#include <nccl.h>

namespace legate::detail {

void show_progress(const Legion::Task* task, Legion::Context ctx, Legion::Runtime* runtime);

}  // namespace legate::detail

namespace legate::comm::nccl {

namespace {

struct Payload {
  uint64_t field0;
  uint64_t field1;
};
}  // namespace

#define CHECK_NCCL(...)                      \
  do {                                       \
    const ncclResult_t result = __VA_ARGS__; \
    check_nccl(result, __FILE__, __LINE__);  \
  } while (false)

inline void check_nccl(ncclResult_t error, const char* file, int line)
{
  if (error != ncclSuccess) {
    static_cast<void>(fprintf(stderr,
                              "Internal NCCL failure with error %s in file %s at line %d\n",
                              ncclGetErrorString(error),
                              file,
                              line));
    exit(error);
  }
}

class Factory final : public detail::CommunicatorFactory {
 public:
  Factory(const detail::Library* core_library);

  [[nodiscard]] bool needs_barrier() const override;
  [[nodiscard]] bool is_supported_target(mapping::TaskTarget target) const override;

 protected:
  Legion::FutureMap initialize(const mapping::detail::Machine& machine,
                               uint32_t num_tasks) override;
  void finalize(const mapping::detail::Machine& machine,
                uint32_t num_tasks,
                const Legion::FutureMap& communicator) override;

 private:
  const detail::Library* core_library_;
};

Factory::Factory(const detail::Library* core_library) : core_library_(core_library) {}

bool Factory::needs_barrier() const { return legate::comm::nccl::needs_barrier(); }

bool Factory::is_supported_target(mapping::TaskTarget target) const
{
  return target == mapping::TaskTarget::GPU;
}

Legion::FutureMap Factory::initialize(const mapping::detail::Machine& machine, uint32_t num_tasks)
{
  Domain launch_domain(Rect<1>(Point<1>(0), Point<1>(static_cast<int64_t>(num_tasks) - 1)));

  // Create a communicator ID
  detail::TaskLauncher init_nccl_id_launcher(
    core_library_, machine, LEGATE_CORE_INIT_NCCL_ID_TASK_ID, LEGATE_GPU_VARIANT);
  init_nccl_id_launcher.set_side_effect(true);
  auto nccl_id = init_nccl_id_launcher.execute_single();

  // Then create the communicators on participating GPUs
  detail::TaskLauncher init_nccl_launcher(
    core_library_, machine, LEGATE_CORE_INIT_NCCL_TASK_ID, LEGATE_GPU_VARIANT);
  init_nccl_launcher.add_future(nccl_id);
  init_nccl_launcher.set_concurrent(true);
  return init_nccl_launcher.execute(launch_domain);
}

void Factory::finalize(const mapping::detail::Machine& machine,
                       uint32_t num_tasks,
                       const Legion::FutureMap& communicator)
{
  Domain launch_domain(Rect<1>(Point<1>(0), Point<1>(static_cast<int64_t>(num_tasks) - 1)));

  detail::TaskLauncher launcher(
    core_library_, machine, LEGATE_CORE_FINALIZE_NCCL_TASK_ID, LEGATE_GPU_VARIANT);
  launcher.set_concurrent(true);
  launcher.add_future_map(communicator);
  launcher.execute(launch_domain);
}

namespace {

ncclUniqueId init_nccl_id(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& regions,
                          Legion::Context context,
                          Legion::Runtime* runtime)
{
  const legate::nvtx::Range auto_range{"core::comm::nccl::init_id"};

  legate::detail::show_progress(task, context, runtime);

  ncclUniqueId id;
  CHECK_NCCL(ncclGetUniqueId(&id));

  return id;
}

ncclComm_t* init_nccl(const Legion::Task* task,
                      const std::vector<Legion::PhysicalRegion>& regions,
                      Legion::Context context,
                      Legion::Runtime* runtime)
{
  const legate::nvtx::Range auto_range{"core::comm::nccl::init"};

  legate::detail::show_progress(task, context, runtime);

  assert(task->futures.size() == 1);

  auto id   = task->futures[0].get_result<ncclUniqueId>();
  auto comm = std::make_unique<ncclComm_t>();

  auto num_ranks = task->index_domain.get_volume();
  auto rank_id   = task->index_point[0];

  auto ts_init_start = std::chrono::high_resolution_clock::now();
  CHECK_NCCL(ncclGroupStart());
  CHECK_NCCL(ncclCommInitRank(comm.get(), num_ranks, id, rank_id));
  CHECK_NCCL(ncclGroupEnd());
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

  auto stream = cuda::StreamPool::get_stream_pool().get_stream();

  // Perform a warm-up all-to-all

  cudaEvent_t ev_start, ev_end_all_to_all, ev_end_all_gather;
  CHECK_CUDA(cudaEventCreate(&ev_start));
  CHECK_CUDA(cudaEventCreate(&ev_end_all_to_all));
  CHECK_CUDA(cudaEventCreate(&ev_end_all_gather));

  auto src_buffer = create_buffer<Payload>(num_ranks, Memory::Kind::GPU_FB_MEM);
  auto tgt_buffer = create_buffer<Payload>(num_ranks, Memory::Kind::GPU_FB_MEM);

  CHECK_CUDA(cudaEventRecord(ev_start, stream));

  CHECK_NCCL(ncclGroupStart());
  for (std::size_t idx = 0; idx < num_ranks; ++idx) {
    CHECK_NCCL(ncclSend(src_buffer.ptr(0), sizeof(Payload), ncclInt8, idx, *comm, stream));
    CHECK_NCCL(ncclRecv(tgt_buffer.ptr(0), sizeof(Payload), ncclInt8, idx, *comm, stream));
  }
  CHECK_NCCL(ncclGroupEnd());

  CHECK_CUDA(cudaEventRecord(ev_end_all_to_all, stream));

  CHECK_NCCL(ncclAllGather(src_buffer.ptr(0), tgt_buffer.ptr(0), 1, ncclUint64, *comm, stream));

  CHECK_CUDA(cudaEventRecord(ev_end_all_gather, stream));

  CHECK_CUDA(cudaEventSynchronize(ev_end_all_gather));

  float time_all_to_all = 0.;
  float time_all_gather = 0.;
  CHECK_CUDA(cudaEventElapsedTime(&time_all_to_all, ev_start, ev_end_all_to_all));
  CHECK_CUDA(cudaEventElapsedTime(&time_all_gather, ev_end_all_to_all, ev_end_all_gather));

  if (0 == rank_id) {
    legate::detail::log_legate().debug(
      "NCCL warm-up took %f ms (all-to-all: %f ms, all-gather: %f ms)",
      time_all_to_all + time_all_gather,
      time_all_to_all,
      time_all_gather);
  }

  return comm.release();
}

void finalize_nccl(const Legion::Task* task,
                   const std::vector<Legion::PhysicalRegion>& regions,
                   Legion::Context context,
                   Legion::Runtime* runtime)
{
  const legate::nvtx::Range auto_range{"core::comm::nccl::finalize"};

  legate::detail::show_progress(task, context, runtime);

  assert(task->futures.size() == 1);
  auto comm = task->futures[0].get_result<ncclComm_t*>();
  CHECK_NCCL(ncclCommDestroy(*comm));
  delete comm;
}

}  // namespace

void register_tasks(Legion::Runtime* runtime, const detail::Library* core_library)
{
  auto init_nccl_id_task_id          = core_library->get_task_id(LEGATE_CORE_INIT_NCCL_ID_TASK_ID);
  const char* init_nccl_id_task_name = "core::comm::nccl::init_id";
  runtime->attach_name(
    init_nccl_id_task_id, init_nccl_id_task_name, false /*mutable*/, true /*local only*/);

  auto init_nccl_task_id          = core_library->get_task_id(LEGATE_CORE_INIT_NCCL_TASK_ID);
  const char* init_nccl_task_name = "core::comm::nccl::init";
  runtime->attach_name(
    init_nccl_task_id, init_nccl_task_name, false /*mutable*/, true /*local only*/);

  auto finalize_nccl_task_id = core_library->get_task_id(LEGATE_CORE_FINALIZE_NCCL_TASK_ID);
  const char* finalize_nccl_task_name = "core::comm::nccl::finalize";
  runtime->attach_name(
    finalize_nccl_task_id, finalize_nccl_task_name, false /*mutable*/, true /*local only*/);

  auto make_registrar = [&](auto task_id, auto* task_name, auto proc_kind) {
    Legion::TaskVariantRegistrar registrar(task_id, task_name);
    registrar.add_constraint(Legion::ProcessorConstraint(proc_kind));
    registrar.set_leaf(true);
    registrar.global_registration = false;
    return registrar;
  };

  // Register the task variants
  {
    auto registrar =
      make_registrar(init_nccl_id_task_id, init_nccl_id_task_name, Processor::TOC_PROC);
    runtime->register_task_variant<ncclUniqueId, init_nccl_id>(registrar, LEGATE_GPU_VARIANT);
  }
  {
    auto registrar = make_registrar(init_nccl_task_id, init_nccl_task_name, Processor::TOC_PROC);
    runtime->register_task_variant<ncclComm_t*, init_nccl>(registrar, LEGATE_GPU_VARIANT);
  }
  {
    auto registrar =
      make_registrar(finalize_nccl_task_id, finalize_nccl_task_name, Processor::TOC_PROC);
    runtime->register_task_variant<finalize_nccl>(registrar, LEGATE_GPU_VARIANT);
  }
}

bool needs_barrier()
{
  // Blocking communications in NCCL violate CUDA's (undocumented) concurrent forward progress
  // requirements and no CUDA drivers that have released are safe from this. Until either CUDA
  // or NCCL is fixed, we will always insert a barrier at the beginning of every NCCL task.
  return true;
}

void register_factory(const detail::Library* core_library)
{
  auto* comm_mgr = detail::Runtime::get_runtime()->communicator_manager();
  comm_mgr->register_factory("nccl", std::make_unique<Factory>(core_library));
}

}  // namespace legate::comm::nccl
