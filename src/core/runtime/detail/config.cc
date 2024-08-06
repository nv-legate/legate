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

#include "core/runtime/detail/config.h"

#include "core/utilities/env.h"

namespace legate::detail {

/*static*/ void Config::reset_() noexcept
{
  Config::show_progress_requested    = false;
  Config::use_empty_task             = false;
  Config::synchronize_stream_view    = false;
  Config::log_mapping_decisions      = false;
  Config::log_partitioning_decisions = false;
  Config::has_socket_mem             = false;
  Config::warmup_nccl                = false;
}

/*static*/ void Config::parse()
{
  try {
    Config::show_progress_requested    = LEGATE_SHOW_PROGRESS.get(/* default_value = */ false);
    Config::use_empty_task             = LEGATE_EMPTY_TASK.get(/* default_value = */ false);
    Config::synchronize_stream_view    = LEGATE_SYNC_STREAM_VIEW.get(/* default_value = */ false);
    Config::log_mapping_decisions      = LEGATE_LOG_MAPPING.get(/* default_value = */ false);
    Config::log_partitioning_decisions = LEGATE_LOG_PARTITIONING.get(/* default_value = */ false);
    Config::warmup_nccl                = LEGATE_WARMUP_NCCL.get(/* default_value = */ false);
  } catch (...) {
    Config::reset_();
    throw;
  }
}

}  // namespace legate::detail
