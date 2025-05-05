/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/comm_cal.h>

#include <legate/comm/coll.h>
#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/cuda/detail/nvtx.h>
#include <legate/data/buffer.h>
#include <legate/operation/detail/task_launcher.h>
#include <legate/runtime/detail/communicator_manager.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/runtime/runtime.h>
#include <legate/task/detail/legion_task.h>
#include <legate/task/task_config.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/typedefs.h>

#include <cal.h>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace legate::detail {

void show_progress(const Legion::Task* task, Legion::Context ctx, Legion::Runtime* runtime);

namespace comm::cal {

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
    default: return "Unknown error code";  // legate-lint: no-switch-default
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
    std::exit(error);
  }
}

[[nodiscard]] calError_t allgather(
  void* src_buf, void* recv_buf, std::size_t size, void* data, void** request)
{
  // this is sync!
  legate::comm::coll::collAllgather(src_buf,
                                    recv_buf,
                                    static_cast<int>(size),
                                    legate::comm::coll::CollDataType::CollInt8,
                                    reinterpret_cast<legate::comm::coll::CollComm>(data));

  // some dummy request
  static auto dummy = 0;

  *request = static_cast<void*>(&dummy);
  return CAL_OK;
}

}  // namespace

class Init : public detail::LegionTask<Init> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{CoreTask::INIT_CAL}};

  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);

  [[nodiscard]] static cal_comm_t gpu_variant(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& /*regions*/,
    Legion::Context context,
    Legion::Runtime* runtime)
  {
    const nvtx3::scoped_range auto_range{task_name_().data()};

    legate::detail::show_progress(task, context, runtime);

    LEGATE_CHECK(task->futures.size() == 1);
    auto* cpu_comm = task->futures[0].get_result<legate::comm::coll::CollComm>();

    /* Create communicator */
    cal_comm_t cal_comm = nullptr;
    cal_comm_create_params_t params{};

    params.allgather    = allgather;
    params.req_test     = [](void*) { return CAL_OK; };
    params.req_free     = [](void*) { return CAL_OK; };
    params.data         = static_cast<void*>(cpu_comm);
    params.rank         = task->index_point[0];
    params.nranks       = task->index_domain.get_volume();
    params.local_device = cuda::detail::get_cuda_driver_api()->ctx_get_device();

    CHECK_CAL(cal_comm_create(params, &cal_comm));

    return cal_comm;
  }
};

class Finalize : public detail::LegionTask<Finalize> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{CoreTask::FINALIZE_CAL}};

  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_concurrent(true);

  static void gpu_variant(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& /*regions*/,
                          Legion::Context context,
                          Legion::Runtime* runtime)
  {
    const nvtx3::scoped_range auto_range{task_name_().data()};

    legate::detail::show_progress(task, context, runtime);

    LEGATE_CHECK(task->futures.size() == 1);
    auto comm = task->futures[0].get_result<cal_comm_t>();
    CHECK_CAL(cal_comm_destroy(comm));
  }
};

class Factory final : public detail::CommunicatorFactory {
 public:
  explicit Factory(const detail::Library& core_library);

  [[nodiscard]] bool needs_barrier() const override;
  [[nodiscard]] bool is_supported_target(mapping::TaskTarget target) const override;

 private:
  [[nodiscard]] static Domain make_launch_domain_(std::uint32_t num_tasks);
  [[nodiscard]] Legion::FutureMap initialize_(const mapping::detail::Machine& machine,
                                              std::uint32_t num_tasks) override;
  void finalize_(const mapping::detail::Machine& machine,
                 std::uint32_t num_tasks,
                 const Legion::FutureMap& communicator) override;

  const detail::Library* core_library_{};
};

Factory::Factory(const detail::Library& core_library) : core_library_{&core_library} {}

bool Factory::needs_barrier() const { return true; }

bool Factory::is_supported_target(mapping::TaskTarget target) const
{
  return target == mapping::TaskTarget::GPU;
}

Domain Factory::make_launch_domain_(std::uint32_t num_tasks)
{
  return Rect<1>{Point<1>{0}, Point<1>{static_cast<std::int64_t>(num_tasks) - 1}};
}

Legion::FutureMap Factory::initialize_(const mapping::detail::Machine& machine,
                                       std::uint32_t num_tasks)
{
  const auto launch_domain = make_launch_domain_(num_tasks);
  auto launcher            = detail::TaskLauncher{*core_library_,
                                       machine,
                                       Init::TASK_CONFIG.task_id(),
                                       static_cast<Legion::MappingTagID>(VariantCode::GPU)};

  // add cpu communicator
  auto&& comm_mgr         = detail::Runtime::get_runtime().communicator_manager();
  auto&& cpu_comm_factory = comm_mgr.find_factory("cpu");
  const auto cpu_comm     = cpu_comm_factory.find_or_create(
    mapping::TaskTarget::GPU, machine.processor_range(), launch_domain);

  launcher.set_concurrent(true);
  // Setting this according to the return type on the task variant. Have to do this manually because
  // this launch is using the Legion task launcher directly.
  launcher.set_future_size(sizeof(cal_comm_t));
  launcher.add_future_map(cpu_comm);
  return launcher.execute(launch_domain);
}

void Factory::finalize_(const mapping::detail::Machine& machine,
                        std::uint32_t num_tasks,
                        const Legion::FutureMap& communicator)
{
  const auto launch_domain = make_launch_domain_(num_tasks);
  auto launcher            = detail::TaskLauncher{*core_library_,
                                       machine,
                                       Finalize::TASK_CONFIG.task_id(),
                                       static_cast<Legion::MappingTagID>(VariantCode::GPU)};

  launcher.set_concurrent(true);
  launcher.add_future_map(communicator);
  launcher.execute(launch_domain);
}

void register_tasks(detail::Library& core_library)
{
  Init::register_variants(legate::Library{&core_library});
  Finalize::register_variants(legate::Library{&core_library});
}

void register_factory(const detail::Library& core_library)
{
  auto&& comm_mgr = detail::Runtime::get_runtime().communicator_manager();

  comm_mgr.register_factory("cal", std::make_unique<Factory>(core_library));
}

}  // namespace comm::cal

}  // namespace legate::detail
