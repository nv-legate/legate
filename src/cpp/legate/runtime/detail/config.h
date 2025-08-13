/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/detail/env_defaults.h>

#include <cstdint>
#include <string>

namespace legate::detail {

#define LEGATE_CONFIG_VAR(__type__, __name__, __initial_value__)                   \
 private:                                                                          \
  __type__ __name__##_ = __initial_value__;                                        \
                                                                                   \
 public:                                                                           \
  [[nodiscard]] const __type__& __name__() const noexcept { return __name__##_; }  \
  void set_##__name__(__type__ value) noexcept { __name__##_ = std::move(value); } \
  static_assert(true)

class LEGATE_PYTHON_EXPORT Config {
 public:
  LEGATE_CONFIG_VAR(bool, auto_config, true);
  LEGATE_CONFIG_VAR(bool, show_config, false);
  LEGATE_CONFIG_VAR(bool, show_progress_requested, false);
  LEGATE_CONFIG_VAR(bool, use_empty_task, false);
  LEGATE_CONFIG_VAR(bool, warmup_nccl, false);
  LEGATE_CONFIG_VAR(bool, enable_inline_task_launch, false);
  LEGATE_CONFIG_VAR(bool, show_mapper_usage, false);
  LEGATE_CONFIG_VAR(std::uint32_t, max_exception_size, LEGATE_MAX_EXCEPTION_SIZE_DEFAULT);
  LEGATE_CONFIG_VAR(std::int64_t, min_cpu_chunk, LEGATE_MIN_CPU_CHUNK_DEFAULT);
  LEGATE_CONFIG_VAR(std::int64_t, min_gpu_chunk, LEGATE_MIN_GPU_CHUNK_DEFAULT);
  LEGATE_CONFIG_VAR(std::int64_t, min_omp_chunk, LEGATE_MIN_OMP_CHUNK_DEFAULT);
  LEGATE_CONFIG_VAR(std::uint32_t, window_size, LEGATE_WINDOW_SIZE_DEFAULT);
  LEGATE_CONFIG_VAR(std::uint32_t, field_reuse_frac, LEGATE_FIELD_REUSE_FRAC_DEFAULT);
  LEGATE_CONFIG_VAR(std::uint32_t, field_reuse_freq, LEGATE_FIELD_REUSE_FREQ_DEFAULT);
  LEGATE_CONFIG_VAR(bool, consensus, LEGATE_CONSENSUS_DEFAULT);
  LEGATE_CONFIG_VAR(bool, disable_mpi, LEGATE_DISABLE_MPI_DEFAULT);
  LEGATE_CONFIG_VAR(bool, io_use_vfd_gds, false);
  LEGATE_CONFIG_VAR(std::int64_t, num_omp_threads, 0);
  LEGATE_CONFIG_VAR(bool, need_cuda, false);
  LEGATE_CONFIG_VAR(bool, need_openmp, false);
  LEGATE_CONFIG_VAR(bool, need_network, false);
  LEGATE_CONFIG_VAR(bool, profile, false);
  LEGATE_CONFIG_VAR(std::string, profile_name, std::string{"legate"});
  LEGATE_CONFIG_VAR(bool, provenance, false);
};

#undef LEGATE_CONFIG_VAR

}  // namespace legate::detail
