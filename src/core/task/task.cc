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

#include "core/task/task.h"

#include "core/runtime/detail/runtime.h"
#include "core/runtime/runtime.h"
#include "core/task/detail/return.h"
#include "core/task/detail/task_context.h"
#include "core/task/exception.h"
#include "core/task/registrar.h"
#include "core/task/task_context.h"
#include "core/utilities/nvtx_help.h"
#include "core/utilities/typedefs.h"

#include "realm/faults.h"

#include <cxxabi.h>
#include <optional>
#include <string_view>

namespace legate::detail {

void show_progress(const Legion::Task* task, Legion::Context ctx, Legion::Runtime* runtime)
{
  if (!Config::show_progress_requested) {
    return;
  }
  const auto exec_proc     = runtime->get_executing_processor(ctx);
  const auto proc_kind_str = [&] {
    switch (const auto kind = exec_proc.kind()) {
      case Processor::LOC_PROC: return "CPU";
      case Processor::TOC_PROC: return "GPU";
      case Processor::OMP_PROC: return "OpenMP";
      default:
        LEGATE_ABORT("Unhandled processor kind: " << legate::traits::detail::to_underlying(kind));
        break;
    }
    return "";
  }();

  std::stringstream point_str;
  const auto& point = task->index_point;
  point_str << point[0];
  for (std::int32_t dim = 1; dim < point.dim; ++dim) {
    point_str << "," << point[dim];
  }

  log_legate().print("%s %s task [%s], pt = (%s), proc = " IDFMT,
                     task->get_task_name(),
                     proc_kind_str,
                     task->get_provenance_string().c_str(),
                     point_str.str().c_str(),
                     exec_proc.id);
}

std::string generate_task_name(const std::type_info& ti)
{
  std::string result;
  int status      = 0;
  char* demangled = abi::__cxa_demangle(ti.name(), nullptr, nullptr, &status);
  result          = demangled;
  std::free(demangled);
  LegateCheck(!status);
  return result;
}

void task_wrapper(VariantImpl variant_impl,
                  LegateVariantCode variant_kind,
                  std::optional<std::string_view> task_name,
                  const void* args,
                  std::size_t arglen,
                  const void* /*userdata*/,
                  std::size_t /*userlen*/,
                  Processor p)
{
  // Legion preamble
  const Legion::Task* task;
  const std::vector<Legion::PhysicalRegion>* regions;
  Legion::Context legion_context;
  Legion::Runtime* runtime;
  Legion::Runtime::legion_task_preamble(args, arglen, p, task, regions, legion_context, runtime);
  const auto get_task_name = [&] {
    return task_name.has_value() ? task_name.value() : task->get_task_name();
  };

  // Cannot use if (LegateDefined(...)) here since nvtx::Range is a RAII class which begins and
  // ends a timer on construction and destruction. It must be in the same lexical scope as the
  // task evaluation!
#if LegateDefined(LEGATE_USE_CUDA)
  std::stringstream ss;
  ss << get_task_name();
  if (!task->get_provenance_string().empty()) {
    ss << " : " + task->get_provenance_string();
  }
  const std::string msg = std::move(ss).str();
  const nvtx::Range auto_range{msg.c_str()};
#endif

  show_progress(task, legion_context, runtime);

  detail::TaskContext context{task, variant_kind, *regions};

  ReturnValues return_values{};

  try {
    const legate::TaskContext ctx{&context};

    if (!Config::use_empty_task) {
      (*variant_impl)(ctx);
    }
    if (auto& excn = ctx.impl()->get_exception(); excn.has_value()) {
      if (context.can_raise_exception()) {
        context.make_all_unbound_stores_empty();
        return_values = context.pack_return_values_with_exception(*excn);
      } else {
        // If a Legate exception is thrown by a task that does not declare any exception,
        // this is a bug in the library that needs to be reported to the developer
        LEGATE_ABORT("Task " << get_task_name().data()
                             << " threw an exception \""
                             // TODO(jfaibussowit): need to extract the actual error message here
                             << "Unknown Python exception"
                             << "\", but the task did not declare any exception.");
      }
    } else {
      return_values = context.pack_return_values();
    }
  } catch (const legate::TaskException& e) {
    if (context.can_raise_exception()) {
      // TODO(jfaibussowit): fix this to not construct this unless we have actually thrown!
      const ReturnedCppException exn{e.index(), e.what()};

      context.make_all_unbound_stores_empty();
      return_values = context.pack_return_values_with_exception(exn);
    } else {
      // If a Legate exception is thrown by a task that does not declare any exception,
      // this is a bug in the library that needs to be reported to the developer
      LEGATE_ABORT("Task " << get_task_name().data() << " threw an exception \"" << e.what()
                           << "\", but the task did not declare any exception.");
    }
  }

  // Legion postamble
  return_values.finalize(legion_context);
}

}  // namespace legate::detail
