/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/task.h>

#include <legate/runtime/detail/config.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/type_traits.h>

#include <cstdint>
#include <ios>
#include <sstream>
#include <string_view>

namespace legate::detail {

void show_progress(const DomainPoint& task_index,
                   std::string_view task_name,
                   VariantCode variant_code,
                   std::string_view provenance)
{
  if (!Runtime::get_runtime().config().show_progress_requested()) {
    return;
  }

  const auto variant_str = [&]() -> std::string_view {
    switch (variant_code) {
      case VariantCode::CPU: return "CPU";
      case VariantCode::GPU: return "GPU";
      case VariantCode::OMP: return "OpenMP";
    }
    LEGATE_ABORT("Unknown variant code: " + std::to_string(to_underlying(variant_code)));
    return "";
  }();

  log_legate().print() << task_name << ' ' << variant_str << " task [" << provenance
                       << "], pt = " << task_index;
}

void show_progress(const DomainPoint& index_point,
                   std::string_view task_name,
                   std::string_view provenance,
                   Legion::Context ctx,
                   Legion::Runtime* runtime)
{
  if (!Runtime::get_runtime().config().show_progress_requested()) {
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
    LEGATE_ABORT("Unhandled processor kind: ", to_underlying(kind));
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
  if (!Runtime::get_runtime().config().show_progress_requested()) {
    return;
  }
  show_progress(task->index_point,
                task->get_task_name(),
                task->get_provenance_string(),
                std::move(ctx),
                runtime);
}

}  // namespace legate::detail
