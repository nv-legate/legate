/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/comm_nccl.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/cuda/detail/cuda_util.h>
#include <legate/cuda/detail/nvtx.h>
#include <legate/data/buffer.h>
#include <legate/runtime/detail/communicator_manager.h>
#include <legate/runtime/detail/config.h>
#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/runtime/runtime.h>
#include <legate/task/task_config.h>
#include <legate/tuning/parallel_policy.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/typedefs.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <nccl.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace legate::detail {

namespace comm::nccl {

namespace {

class Factory final : public detail::CommunicatorFactory {
 public:
  explicit Factory(const detail::Library& core_library);

  [[nodiscard]] bool needs_barrier() const override;
  [[nodiscard]] bool is_supported_target(mapping::TaskTarget target) const override;

 protected:
  [[nodiscard]] Legion::FutureMap initialize_(const mapping::detail::Machine& machine,
                                              std::uint32_t num_tasks) override;
  void finalize_(const mapping::detail::Machine& machine,
                 std::uint32_t num_tasks,
                 const Legion::FutureMap& communicator) override;

  const detail::Library* core_library_{};
};

Factory::Factory(const detail::Library& core_library) : core_library_{&core_library} {}

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

Legion::FutureMap Factory::initialize_(const mapping::detail::Machine&, std::uint32_t)
{
  // TODO(wonchanl): Need a v2 implementation
  return {};
}

void Factory::finalize_(const mapping::detail::Machine&, std::uint32_t, const Legion::FutureMap&)
{
  // TODO(wonchanl): Need a v2 implementation
}

}  // namespace

void register_factory(const detail::Library& core_library)
{
  auto&& comm_mgr = detail::Runtime::get_runtime().communicator_manager();
  comm_mgr.register_factory("nccl", std::make_unique<Factory>(core_library));
}

}  // namespace comm::nccl

}  // namespace legate::detail
