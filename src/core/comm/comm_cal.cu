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

#include "core/comm/coll.h"
#include "core/comm/comm_cal.h"
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#include "core/data/buffer.h"
#include "core/operation/detail/task_launcher.h"
#include "core/runtime/detail/communicator_manager.h"
#include "core/runtime/detail/library.h"
#include "core/runtime/detail/runtime.h"
#include "core/runtime/runtime.h"
#include "core/task/detail/task_context.h"
#include "core/utilities/nvtx_help.h"
#include "core/utilities/typedefs.h"

#include <cal.h>
#include <chrono>
#include <cuda.h>

namespace legate::detail {

void show_progress(const Legion::Task* task, Legion::Context ctx, Legion::Runtime* runtime);

}  // namespace legate::detail

namespace legate::comm::cal {

#define CHECK_CAL(...)                     \
  do {                                     \
    const calError_t result = __VA_ARGS__; \
    check_cal(result, __FILE__, __LINE__); \
  } while (false)

namespace {

const char* cal_get_error_message(calError_t error)
{
  switch (error) {
    case CAL_OK: return "Success";
    case CAL_ERROR_INPROGRESS: return "Request is in progress";
    case CAL_ERROR: return "Generic error";
    case CAL_ERROR_INVALID_PARAMETER: return "Invalid parameter to the interface function";
    case CAL_ERROR_INTERNAL: return "Internal error";
    case CAL_ERROR_CUDA: return "Error in CUDA runtime/driver API";
    case CAL_ERROR_UCC: return "Error in UCC call";
    case CAL_ERROR_NOT_SUPPORTED: return "Requested configuration or parameters are not supported";
    default: return "Unknown error code";
  }
}

void check_cal(calError_t error, const char* file, int line)
{
  if (error != CAL_OK) {
    static_cast<void>(fprintf(stderr,
                              "Internal CAL failure with error %d (%s) in file %s at line %d\n",
                              error,
                              cal_get_error_message(error),
                              file,
                              line));
    exit(error);
  }
}

}  // namespace

class Factory final : public detail::CommunicatorFactory {
 public:
  explicit Factory(const detail::Library* core_library);

 public:
  [[nodiscard]] bool needs_barrier() const override;
  [[nodiscard]] bool is_supported_target(mapping::TaskTarget target) const override;

 protected:
  [[nodiscard]] Legion::FutureMap initialize(const mapping::detail::Machine& machine,
                                             uint32_t num_tasks) override;
  void finalize(const mapping::detail::Machine& machine,
                uint32_t num_tasks,
                const Legion::FutureMap& communicator) override;

 private:
  const detail::Library* core_library_{};
};

Factory::Factory(const detail::Library* core_library) : core_library_{core_library} {}

bool Factory::needs_barrier() const { return true; }

bool Factory::is_supported_target(mapping::TaskTarget target) const
{
  return target == mapping::TaskTarget::GPU;
}

Legion::FutureMap Factory::initialize(const mapping::detail::Machine& machine, uint32_t num_tasks)
{
  Domain launch_domain{Rect<1>{Point<1>{0}, Point<1>{static_cast<int64_t>(num_tasks) - 1}}};

  detail::TaskLauncher init_cal_launcher{
    core_library_, machine, LEGATE_CORE_INIT_CAL_TASK_ID, LEGATE_GPU_VARIANT};
  init_cal_launcher.set_concurrent(true);

  // add cpu communicator
  auto* comm_mgr         = detail::Runtime::get_runtime()->communicator_manager();
  auto* cpu_comm_factory = comm_mgr->find_factory("cpu");
  auto cpu_comm          = cpu_comm_factory->find_or_create(
    mapping::TaskTarget::GPU, machine.processor_range(), launch_domain);
  init_cal_launcher.add_future_map(cpu_comm);

  return init_cal_launcher.execute(launch_domain);
}

void Factory::finalize(const mapping::detail::Machine& machine,
                       uint32_t num_tasks,
                       const Legion::FutureMap& communicator)
{
  Domain launch_domain{Rect<1>{Point<1>{0}, Point<1>{static_cast<int64_t>(num_tasks) - 1}}};

  detail::TaskLauncher launcher{
    core_library_, machine, LEGATE_CORE_FINALIZE_CAL_TASK_ID, LEGATE_GPU_VARIANT};
  launcher.set_concurrent(true);
  launcher.add_future_map(communicator);
  launcher.execute(launch_domain);
}

namespace {

[[nodiscard]] calError_t allgather(
  void* src_buf, void* recv_buf, size_t size, void* data, void** request)
{
  // this is sync!
  auto res = comm::coll::collAllgather(src_buf,
                                       recv_buf,
                                       size,
                                       comm::coll::CollDataType::CollInt8,
                                       reinterpret_cast<comm::coll::CollComm>(data));
  if (res != 0) {
    return CAL_ERROR;
  }

  // some dummy request
  auto dummy = new calError_t{};
  *request   = static_cast<void*>(dummy);

  return CAL_OK;
}

[[nodiscard]] calError_t request_test(void*) { return CAL_OK; }
[[nodiscard]] calError_t request_free(void* request)
{
  delete reinterpret_cast<calError_t*>(request);
  return CAL_OK;
}

[[nodiscard]] cal_comm_t init_cal(const Legion::Task* task,
                                  const std::vector<Legion::PhysicalRegion>& /*regions*/,
                                  Legion::Context context,
                                  Legion::Runtime* runtime)
{
  legate::nvtx::Range auto_range{"core::comm::cal::init"};

  legate::detail::show_progress(task, context, runtime);

  auto rank      = task->index_point[0];
  auto num_ranks = task->index_domain.get_volume();

  LegateCheck(task->futures.size() == 1);
  auto cpu_comm = task->futures[0].get_result<comm::coll::CollComm>();

  int device = -1;
  CHECK_CUDA(cudaGetDevice(&device));

  /* Create communicator */
  cal_comm_t cal_comm = nullptr;
  cal_comm_create_params_t params;
  params.allgather    = allgather;
  params.req_test     = request_test;
  params.req_free     = request_free;
  params.data         = reinterpret_cast<void*>(cpu_comm);
  params.rank         = rank;
  params.nranks       = num_ranks;
  params.local_device = device;

  CHECK_CAL(cal_comm_create(params, &cal_comm));

  return cal_comm;
}

void finalize_cal(const Legion::Task* task,
                  const std::vector<Legion::PhysicalRegion>& /*regions*/,
                  Legion::Context context,
                  Legion::Runtime* runtime)
{
  legate::nvtx::Range auto_range{"core::comm::cal::finalize"};

  legate::detail::show_progress(task, context, runtime);

  LegateCheck(task->futures.size() == 1);
  auto comm = task->futures[0].get_result<cal_comm_t>();
  CHECK_CAL(cal_comm_destroy(comm));
}

}  // namespace

void register_tasks(const detail::Library* core_library)
{
  const auto runtime    = Legion::Runtime::get_runtime();
  auto init_cal_task_id = core_library->get_task_id(LEGATE_CORE_INIT_CAL_TASK_ID);
  constexpr const char* init_cal_task_name = "core::comm::cal::init";
  runtime->attach_name(
    init_cal_task_id, init_cal_task_name, false /*mutable*/, true /*local only*/);

  auto finalize_cal_task_id = core_library->get_task_id(LEGATE_CORE_FINALIZE_CAL_TASK_ID);
  constexpr const char* finalize_cal_task_name = "core::comm::cal::finalize";
  runtime->attach_name(
    finalize_cal_task_id, finalize_cal_task_name, false /*mutable*/, true /*local only*/);

  auto make_registrar = [&](auto task_id, auto* task_name, auto proc_kind) {
    Legion::TaskVariantRegistrar registrar{task_id, task_name};

    registrar.add_constraint(Legion::ProcessorConstraint{proc_kind});
    registrar.set_leaf(true);
    registrar.global_registration = false;
    return registrar;
  };

  // Register the task variants
  {
    auto registrar = make_registrar(init_cal_task_id, init_cal_task_name, Processor::TOC_PROC);
    runtime->register_task_variant<cal_comm_t, init_cal>(registrar, LEGATE_GPU_VARIANT);
  }
  {
    auto registrar =
      make_registrar(finalize_cal_task_id, finalize_cal_task_name, Processor::TOC_PROC);
    runtime->register_task_variant<finalize_cal>(registrar, LEGATE_GPU_VARIANT);
  }
}

void register_factory(const detail::Library* core_library)
{
  auto* comm_mgr = detail::Runtime::get_runtime()->communicator_manager();
  comm_mgr->register_factory("cal", std::make_unique<Factory>(core_library));
}

}  // namespace legate::comm::cal
