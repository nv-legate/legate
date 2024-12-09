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

#pragma once

#include <legate_defines.h>

#include "legate/runtime/detail/config.h"
#include "legate/task/detail/task.h"
#include "legate/task/detail/task_context.h"
#include "legate/task/exception.h"
#include "legate/utilities/abort.h"
#include "legate/utilities/macros.h"
#include <legate/utilities/compiler.h>

#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace legate::detail::task_detail {

template <typename F>
[[nodiscard]] std::optional<ReturnedException> task_body(legate::TaskContext ctx,
                                                         VariantImpl variant_impl,
                                                         F&& get_task_name)
{
  const auto handle_raised_exception = [&](ReturnedException exn,
                                           std::string_view exn_text) -> ReturnedException {
    if (ctx.can_raise_exception()) {
      ctx.impl()->make_all_unbound_stores_empty();
      return exn;
    }
    // If a Legate exception is thrown by a task that does not declare any exception,
    // this is a bug in the library that needs to be reported to the developer
    LEGATE_ABORT("Task \"",
                 get_task_name(),
                 "\" threw an exception \"",
                 exn_text,
                 "\", but the task did not declare any exception.");
  };
  const auto report_unsupported_exception_type = [&](std::string_view exn_type,
                                                     std::string_view exn_message) {
    LEGATE_ABORT("Task \"",
                 get_task_name(),
                 "\" threw an unsupported exception type (",
                 exn_type,
                 "): \"",
                 exn_message,
                 "\". Legate tasks may only "
                 "throw instances "  // legate-lint: no-trace
                 "of legate::TaskException.");
  };

  try {
    if (!Config::use_empty_task) {
      (*variant_impl)(ctx);
    }

    if (auto&& excn = ctx.impl()->get_exception(); excn.has_value()) {
      // TODO(jfaibussowit): need to extract the actual error message here
      LEGATE_ASSERT(excn->kind() == ExceptionKind::PYTHON);
      return handle_raised_exception(*std::move(excn), "Unknown Python Exception");
    }
  } catch (const legate::TaskException& e) {
    return handle_raised_exception(ReturnedCppException{e.index(), e.what()}, e.what());
  } catch (const std::exception& e) {
    report_unsupported_exception_type(demangle_type(typeid(e)), e.what());
  } catch (...) {
    report_unsupported_exception_type("unknown exception type", "unknown exception");
  }
  return std::nullopt;
}

template <typename T, typename U>
[[nodiscard]] nvtx3::scoped_range make_nvtx_range(T&& get_task_name, U&& get_provenance)
{
  std::string msg;

  if (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    auto&& task_name   = get_task_name();
    auto&& provenance  = get_provenance();
    constexpr auto sep = std::string_view{" : "};

    msg.reserve(task_name.size() + sep.size() + provenance.size() + 1);
    msg += task_name;
    if (!provenance.empty()) {
      msg += sep;
      msg += provenance;
    }
  }
  return nvtx3::scoped_range{msg.c_str()};
}

}  // namespace legate::detail::task_detail
