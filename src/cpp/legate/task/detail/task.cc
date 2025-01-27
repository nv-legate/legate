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

#include <legate/task/detail/task.h>

#include <legate/runtime/detail/config.h>

#include <cstdint>
#include <ios>
#include <sstream>
#include <string_view>

namespace legate::detail {

void show_progress(const DomainPoint& index_point,
                   std::string_view task_name,
                   std::string_view provenance,
                   Legion::Context ctx,
                   Legion::Runtime* runtime)
{
  if (!Config::show_progress_requested) {
    return;
  }

  const auto exec_proc     = runtime->get_executing_processor(ctx);
  const auto proc_kind_str = [&]() -> std::string_view {
    const auto kind = exec_proc.kind();

    switch (kind) {
      case Processor::Kind::LOC_PROC: return "CPU";
      case Processor::Kind::TOC_PROC: return "GPU";
      case Processor::Kind::OMP_PROC: return "OpenMP";
      case Processor::Kind::PY_PROC: return "Python";
      case Processor::Kind::NO_KIND: [[fallthrough]];
      case Processor::Kind::UTIL_PROC: [[fallthrough]];
      case Processor::Kind::IO_PROC: [[fallthrough]];
      case Processor::Kind::PROC_GROUP: [[fallthrough]];
      case Processor::Kind::PROC_SET: break;
    }
    LEGATE_ABORT("Unhandled processor kind: ", legate::traits::detail::to_underlying(kind));
  }();

  std::stringstream point_str;

  point_str << index_point[0];
  for (std::int32_t dim = 1; dim < index_point.dim; ++dim) {
    point_str << ',' << index_point[dim];
  }

  log_legate().print() << task_name << ' ' << proc_kind_str << " task [" << provenance
                       << "], pt = (" << std::move(point_str).str() << "), proc = " << std::hex
                       << exec_proc.id;
}

void show_progress(const Legion::Task* task, Legion::Context ctx, Legion::Runtime* runtime)
{
  if (!Config::show_progress_requested) {
    return;
  }
  show_progress(task->index_point,
                task->get_task_name(),
                task->get_provenance_string(),
                std::move(ctx),
                runtime);
}

}  // namespace legate::detail
