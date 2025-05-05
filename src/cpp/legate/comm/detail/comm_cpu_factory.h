/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/mapping.h>
#include <legate/runtime/detail/communicator_manager.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>

namespace legate::detail {

class Library;

}  // namespace legate::detail

namespace legate::detail::comm::cpu {

template <typename InitTaskT, typename InitMappingTaskT, typename FinalizeTaskT>
class Factory final : public detail::CommunicatorFactory {
 public:
  using init_task_type         = InitTaskT;
  using init_mapping_task_type = InitMappingTaskT;
  using finalize_task_type     = FinalizeTaskT;

  explicit Factory(const detail::Library& core_library);

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

}  // namespace legate::detail::comm::cpu

#include <legate/comm/detail/comm_cpu_factory.inl>
