/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/config.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/env.h>

namespace legate::detail {

namespace {

template <typename T>
void parse_value(const EnvironmentVariable<T>& env_var, T* config_var)
{
  *config_var = env_var.get(/* default_value */ *config_var);
}

template <typename T, typename U = T>
void parse_value_test_default(const EnvironmentVariable<T>& env_var,
                              const U& test_val,
                              T* config_var)
{
  *config_var = env_var.get(/* default_value */ *config_var, /* test_val */ test_val);
}

[[nodiscard]] bool single_node_job()
{
  constexpr EnvironmentVariable<std::uint32_t> OMPI_COMM_WORLD_SIZE{"OMPI_COMM_WORLD_SIZE"};
  constexpr EnvironmentVariable<std::uint32_t> MV2_COMM_WORLD_SIZE{"MV2_COMM_WORLD_SIZE"};
  constexpr EnvironmentVariable<std::uint32_t> SLURM_NTASKS{"SLURM_NTASKS"};

  return OMPI_COMM_WORLD_SIZE.get(/* default_value */ 1) == 1 &&
         MV2_COMM_WORLD_SIZE.get(/* default_value */ 1) == 1 &&
         SLURM_NTASKS.get(/* default_vaule */ 1) == 1 &&
         REALM_UCP_BOOTSTRAP_MODE.get(/* default_value*/ "") != "p2p";
}

}  // namespace

void Config::parse()
{
  LEGATE_CHECK(!parsed());
  try {
    parse_value(LEGATE_AUTO_CONFIG, &auto_config_);
    parse_value(LEGATE_SHOW_CONFIG, &show_config_);
    parse_value(LEGATE_SHOW_PROGRESS, &show_progress_requested_);
    parse_value(LEGATE_EMPTY_TASK, &use_empty_task_);
    parse_value(LEGATE_SYNC_STREAM_VIEW, &synchronize_stream_view_);
    parse_value(LEGATE_LOG_MAPPING, &log_mapping_decisions_);
    parse_value(LEGATE_LOG_PARTITIONING, &log_partitioning_decisions_);
    parse_value(LEGATE_WARMUP_NCCL, &warmup_nccl_);
    parse_value(experimental::LEGATE_INLINE_TASK_LAUNCH, &enable_inline_task_launch_);
    parse_value(LEGATE_SHOW_USAGE, &show_mapper_usage_);
    parse_value_test_default(
      LEGATE_MAX_EXCEPTION_SIZE, LEGATE_MAX_EXCEPTION_SIZE_TEST, &max_exception_size_);
    parse_value_test_default(LEGATE_MIN_CPU_CHUNK, LEGATE_MIN_CPU_CHUNK_TEST, &min_cpu_chunk_);
    parse_value_test_default(LEGATE_MIN_GPU_CHUNK, LEGATE_MIN_GPU_CHUNK_TEST, &min_gpu_chunk_);
    parse_value_test_default(LEGATE_MIN_OMP_CHUNK, LEGATE_MIN_OMP_CHUNK_TEST, &min_omp_chunk_);
    parse_value_test_default(LEGATE_WINDOW_SIZE, LEGATE_WINDOW_SIZE_TEST, &window_size_);
    parse_value_test_default(
      LEGATE_FIELD_REUSE_FRAC, LEGATE_FIELD_REUSE_FRAC_TEST, &field_reuse_frac_);
    parse_value_test_default(
      LEGATE_FIELD_REUSE_FREQ, LEGATE_FIELD_REUSE_FREQ_TEST, &field_reuse_freq_);
    parse_value_test_default(LEGATE_CONSENSUS, LEGATE_CONSENSUS_TEST, &consensus_);
    parse_value_test_default(LEGATE_DISABLE_MPI, LEGATE_DISABLE_MPI_TEST, &disable_mpi_);
    parse_value(LEGATE_IO_USE_VFD_GDS, &io_use_vfd_gds_);
    set_need_network(!single_node_job());
  } catch (...) {
    constexpr Config DEFAULT_CONFIG{};

    *this = DEFAULT_CONFIG;
    throw;
  }
  parsed_ = true;
}

bool Config::parsed() const noexcept { return parsed_; }

// ==========================================================================================

namespace {

Config the_config{};

}  // namespace

/* static */ const Config& Config::get_config() noexcept { return the_config; }

/* static */ Config& Config::get_config_mut() noexcept { return the_config; }

}  // namespace legate::detail
