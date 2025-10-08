/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/detail/argument_parsing/argument.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace legate::detail {

/**
 * @brief A structure containing the various command-line flags set by Legate.
 */
class ParsedArgs {
 public:
  Argument<bool> auto_config;
  Argument<bool> show_config;
  Argument<bool> show_progress;
  Argument<bool> empty_task;
  Argument<bool> warmup_nccl;
  Argument<bool> inline_task_launch;
  Argument<bool> show_usage;
  Argument<std::uint32_t> max_exception_size;
  Argument<std::int64_t> min_cpu_chunk;
  Argument<std::int64_t> min_gpu_chunk;
  Argument<std::int64_t> min_omp_chunk;
  Argument<std::uint32_t> window_size;
  Argument<std::uint32_t> field_reuse_frac;
  Argument<std::uint32_t> field_reuse_freq;
  Argument<bool> consensus;
  Argument<bool> disable_mpi;
  Argument<bool> io_use_vfd_gds;
  Argument<std::int32_t> cpus;
  Argument<std::int32_t> gpus;
  Argument<std::int32_t> omps;
  Argument<std::int32_t> ompthreads;
  Argument<std::int32_t> util;
  Argument<Scaled<std::int64_t>> sysmem;
  Argument<Scaled<std::int64_t>> numamem;
  Argument<Scaled<std::int64_t>> fbmem;
  Argument<Scaled<std::int64_t>> zcmem;
  Argument<Scaled<std::int64_t>> regmem;
  Argument<bool> profile;
  Argument<std::string> profile_name;
  Argument<bool> provenance;
  Argument<std::string> log_levels;
  Argument<std::filesystem::path> log_dir;
  Argument<bool> log_to_file;
  Argument<bool> freeze_on_error;
  Argument<std::string> cuda_driver_path;

  /**
   * @brief Return a summary of the current configuration options suitable for printing.
   *
   * @return The summary.
   */
  [[nodiscard]] std::string config_summary() const;
};

/**
 * @brief Parse the given command-line flags and return their values.
 *
 * `args` must not be empty.
 *
 * @param args A list of command-line flags.
 *
 * @return The parsed command-line values.
 */
[[nodiscard]] ParsedArgs parse_args(std::vector<std::string> args);

}  // namespace legate::detail
