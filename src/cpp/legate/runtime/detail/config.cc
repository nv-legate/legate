/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/runtime/detail/config.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/env.h>

namespace legate::detail {

/*static*/ void Config::reset_() noexcept
{
  Config::parsed_                    = false;
  Config::auto_config                = true;
  Config::show_config                = false;
  Config::show_progress_requested    = false;
  Config::use_empty_task             = false;
  Config::synchronize_stream_view    = false;
  Config::log_mapping_decisions      = false;
  Config::log_partitioning_decisions = false;
  Config::has_socket_mem             = false;
  Config::warmup_nccl                = false;
  Config::enable_inline_task_launch  = false;
  Config::num_omp_threads            = 0;
  Config::show_mapper_usage          = false;
  Config::need_cuda                  = false;
  Config::need_openmp                = false;
  Config::need_network               = false;
  Config::max_exception_size         = LEGATE_MAX_EXCEPTION_SIZE_DEFAULT;
  Config::min_cpu_chunk              = LEGATE_MIN_CPU_CHUNK_DEFAULT;
  Config::min_gpu_chunk              = LEGATE_MIN_GPU_CHUNK_DEFAULT;
  Config::min_omp_chunk              = LEGATE_MIN_OMP_CHUNK_DEFAULT;
  Config::window_size                = LEGATE_WINDOW_SIZE_DEFAULT;
  Config::field_reuse_frac           = LEGATE_FIELD_REUSE_FRAC_DEFAULT;
  Config::field_reuse_freq           = LEGATE_FIELD_REUSE_FREQ_DEFAULT;
  Config::consensus                  = LEGATE_CONSENSUS_DEFAULT;
  Config::disable_mpi                = LEGATE_DISABLE_MPI_DEFAULT;
  Config::io_use_vfd_gds             = false;
}

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
  *config_var = env_var.get(/* default_value */ *config_var, test_val);
}

}  // namespace

/*static*/ void Config::parse()
{
  LEGATE_CHECK(!parsed());

  try {
    parse_value(LEGATE_AUTO_CONFIG, &Config::auto_config);
    parse_value(LEGATE_SHOW_CONFIG, &Config::show_config);
    parse_value(LEGATE_SHOW_PROGRESS, &Config::show_progress_requested);
    parse_value(LEGATE_EMPTY_TASK, &Config::use_empty_task);
    parse_value(LEGATE_SYNC_STREAM_VIEW, &Config::synchronize_stream_view);
    parse_value(LEGATE_LOG_MAPPING, &Config::log_mapping_decisions);
    parse_value(LEGATE_LOG_PARTITIONING, &Config::log_partitioning_decisions);
    parse_value(LEGATE_WARMUP_NCCL, &Config::warmup_nccl);
    parse_value(experimental::LEGATE_INLINE_TASK_LAUNCH, &Config::enable_inline_task_launch);
    parse_value(LEGATE_SHOW_USAGE, &Config::show_mapper_usage);
    parse_value(LEGATE_NEED_CUDA, &Config::need_cuda);
    parse_value(LEGATE_NEED_OPENMP, &Config::need_openmp);
    parse_value(LEGATE_NEED_NETWORK, &Config::need_network);
    parse_value_test_default(
      LEGATE_MAX_EXCEPTION_SIZE, LEGATE_MAX_EXCEPTION_SIZE_TEST, &Config::max_exception_size);
    parse_value_test_default(
      LEGATE_MIN_CPU_CHUNK, LEGATE_MIN_CPU_CHUNK_TEST, &Config::min_cpu_chunk);
    parse_value_test_default(
      LEGATE_MIN_GPU_CHUNK, LEGATE_MIN_GPU_CHUNK_TEST, &Config::min_gpu_chunk);
    parse_value_test_default(
      LEGATE_MIN_OMP_CHUNK, LEGATE_MIN_OMP_CHUNK_TEST, &Config::min_omp_chunk);
    parse_value_test_default(LEGATE_WINDOW_SIZE, LEGATE_WINDOW_SIZE_TEST, &Config::window_size);
    parse_value_test_default(
      LEGATE_FIELD_REUSE_FRAC, LEGATE_FIELD_REUSE_FRAC_TEST, &Config::field_reuse_frac);
    parse_value_test_default(
      LEGATE_FIELD_REUSE_FREQ, LEGATE_FIELD_REUSE_FREQ_TEST, &Config::field_reuse_freq);
    parse_value_test_default(LEGATE_CONSENSUS, LEGATE_CONSENSUS_TEST, &Config::consensus);
    parse_value_test_default(LEGATE_DISABLE_MPI, LEGATE_DISABLE_MPI_TEST, &Config::disable_mpi);
    parse_value(LEGATE_IO_USE_VFD_GDS, &Config::io_use_vfd_gds);
  } catch (...) {
    Config::reset_();
    throw;
  }
  Config::parsed_ = true;
}

/*static*/ bool Config::parsed() noexcept { return Config::parsed_; }

}  // namespace legate::detail
