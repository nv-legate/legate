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

#include "core/runtime/detail/config.h"

#include "core/utilities/env.h"
#include "core/utilities/macros.h"

#include <cstdlib>
#include <exception>
#include <iostream>

namespace legate::detail {

/*static*/ void Config::parse()
{
  if (!LEGATE_DEFINED(LEGATE_USE_CUDA) && LEGATE_NEED_CUDA.get(/* default_value = */ false)) {
    // ignore fprintf return values here, we are about to exit anyways
    std::cerr << "[legate.core] Legate was run with GPUs but was not built with GPU support. "
                 "Please install Legate again with the \"--with-cuda\" flag.\n";
    std::exit(1);
  }
  if (!LEGATE_DEFINED(LEGATE_USE_OPENMP) && LEGATE_NEED_OPENMP.get(/* default_value = */ false)) {
    std::cerr
      << "[legate.core] Legate was run with OpenMP enabled, but was not built with OpenMP support. "
         "Please install Legate again with the \"--with-openmp\" flag.\n";
    std::exit(1);
  }
  if (!LEGATE_DEFINED(LEGATE_USE_NETWORK) && LEGATE_NEED_NETWORK.get(/* default_value = */ false)) {
    std::cerr
      << "[legate.core] Legate was run on multiple nodes but was not built with networking "
         "support. Please install Legate again with network support (e.g. \"--with-mpi\" or "
         "\"--with-gasnet\").\n";
    std::exit(1);
  }

  try {
    Config::show_progress_requested    = LEGATE_SHOW_PROGRESS.get(/* default_value = */ false);
    Config::use_empty_task             = LEGATE_EMPTY_TASK.get(/* default_value = */ false);
    Config::synchronize_stream_view    = LEGATE_SYNC_STREAM_VIEW.get(/* default_value = */ false);
    Config::log_mapping_decisions      = LEGATE_LOG_MAPPING.get(/* default_value = */ false);
    Config::log_partitioning_decisions = LEGATE_LOG_PARTITIONING.get(/* default_value = */ false);
    Config::warmup_nccl                = LEGATE_WARMUP_NCCL.get(/* default_value = */ false);
  } catch (const std::exception& e) {
    std::cerr << "[legate.core] " << e.what() << "\n";
    std::exit(1);
  } catch (...) {
    std::cerr << "[legate.core] encountered an unknown error when parsing environment variables\n";
    std::exit(1);
  }
}

}  // namespace legate::detail
