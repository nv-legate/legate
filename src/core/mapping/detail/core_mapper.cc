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

#include "core/mapping/detail/core_mapper.h"

#include "core/comm/comm_cal.h"
#include "core/comm/comm_nccl.h"
#include "core/mapping/detail/machine.h"
#include "core/utilities/env.h"

#include "env_defaults.h"

#include <cstdlib>
#include <vector>

namespace legate::mapping::detail {

// This is a custom mapper implementation that only has to map
// start-up tasks associated with the Legate core, no one else
// should be overriding this mapper so we burry it in here
class CoreMapper final : public Mapper {
 public:
  void set_machine(const legate::mapping::MachineQueryInterface* machine) override;
  [[nodiscard]] legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::TaskTarget>& options) override;
  [[nodiscard]] std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override;
  [[nodiscard]] legate::Scalar tunable_value(legate::TunableID tunable_id) override;

 private:
  const LocalMachine MACHINE{};
  // TODO(wonchanl): Some of these should be moved to legate::detail::Config
  const std::int64_t MIN_GPU_CHUNK{
    LEGATE_MIN_GPU_CHUNK.get(MIN_GPU_CHUNK_DEFAULT, MIN_GPU_CHUNK_TEST)};
  const std::int64_t MIN_CPU_CHUNK{
    LEGATE_MIN_CPU_CHUNK.get(MIN_CPU_CHUNK_DEFAULT, MIN_CPU_CHUNK_TEST)};
  const std::int64_t MIN_OMP_CHUNK{
    LEGATE_MIN_OMP_CHUNK.get(MIN_OMP_CHUNK_DEFAULT, MIN_OMP_CHUNK_TEST)};
  const std::uint32_t WINDOW_SIZE{LEGATE_WINDOW_SIZE.get(WINDOW_SIZE_DEFAULT, WINDOW_SIZE_TEST)};
  const std::uint32_t FIELD_REUSE_FRAC{
    LEGATE_FIELD_REUSE_FRAC.get(FIELD_REUSE_FRAC_DEFAULT, FIELD_REUSE_FRAC_TEST)};
  const std::uint32_t MAX_LRU_LENGTH{
    LEGATE_MAX_LRU_LENGTH.get(MAX_LRU_LENGTH_DEFAULT, MAX_LRU_LENGTH_TEST)};
};

void CoreMapper::set_machine(const legate::mapping::MachineQueryInterface* /*m*/) {}

TaskTarget CoreMapper::task_target(const legate::mapping::Task&,
                                   const std::vector<TaskTarget>& options)
{
  return options.front();
}

std::vector<legate::mapping::StoreMapping> CoreMapper::store_mappings(
  const legate::mapping::Task&, const std::vector<StoreTarget>&)
{
  return {};
}

Scalar CoreMapper::tunable_value(TunableID tunable_id)
{
  switch (tunable_id) {
    case LEGATE_CORE_TUNABLE_TOTAL_CPUS: {
      return Scalar{static_cast<std::int32_t>(MACHINE.total_cpu_count())};  // assume symmetry
    }
    case LEGATE_CORE_TUNABLE_TOTAL_GPUS: {
      return Scalar{static_cast<std::int32_t>(MACHINE.total_gpu_count())};  // assume symmetry
    }
    case LEGATE_CORE_TUNABLE_TOTAL_OMPS: {
      return Scalar{static_cast<std::int32_t>(MACHINE.total_omp_count())};  // assume symmetry
    }
    case LEGATE_CORE_TUNABLE_NUM_NODES: {
      return Scalar{static_cast<std::int32_t>(MACHINE.total_nodes)};
    }
    case LEGATE_CORE_TUNABLE_MIN_SHARD_VOLUME: {
      // TODO(wonchanl): make these profile guided
      if (MACHINE.has_gpus()) {
        // Make sure we can get at least 1M elements on each GPU
        return Scalar{MIN_GPU_CHUNK};
      }
      if (MACHINE.has_omps()) {
        // Make sure we get at least 128K elements on each OpenMP
        return Scalar{MIN_OMP_CHUNK};
      }
      // Make sure we can get at least 8KB elements on each CPU
      return Scalar{MIN_CPU_CHUNK};
    }
    case LEGATE_CORE_TUNABLE_HAS_SOCKET_MEM: {
      return Scalar{MACHINE.has_socket_memory()};
    }
    case LEGATE_CORE_TUNABLE_WINDOW_SIZE: {
      return Scalar{WINDOW_SIZE};
    }
    case LEGATE_CORE_TUNABLE_FIELD_REUSE_SIZE: {
      // Multiply this by the total number of nodes and then scale by the frac
      const auto global_mem_size = [&] {
        if (MACHINE.has_gpus()) {
          return MACHINE.total_frame_buffer_size();
        }
        if (MACHINE.has_socket_memory()) {
          return MACHINE.total_socket_memory_size();
        }
        return MACHINE.system_memory().capacity();
      }();

      return Scalar{global_mem_size / FIELD_REUSE_FRAC};
    }
    case LEGATE_CORE_TUNABLE_MAX_LRU_LENGTH: {
      return Scalar{MAX_LRU_LENGTH};
    }
    default: break;
  }
  // Illegal tunable variable
  LEGATE_ABORT("Illegal tunable variable" << tunable_id);
  return Scalar{0};
}

std::unique_ptr<Mapper> create_core_mapper() { return std::make_unique<CoreMapper>(); }

}  // namespace legate::mapping::detail
