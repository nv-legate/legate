/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/runtime/detail/config.h>
#include <legate/task/detail/task.h>
#include <legate/task/detail/task_context.h>
#include <legate/task/exception.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/compiler.h>
#include <legate/utilities/macros.h>

#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace legate::detail::task_detail {

template <typename F>
[[nodiscard]] std::optional<ReturnedException> task_body(legate::TaskContext ctx,
                                                         VariantImpl variant_impl,
                                                         F&& get_task_name)
{
  constexpr auto EXN_EXPLANATION = std::string_view{
    "If this exception is expected to be thrown (and handled), then you must explicitly "
    "inform the runtime that this task may raise an exception via the AutoTask/ManualTask "
    "throws_exception() member function on task construction. Furthermore, you must wrap the "
    "exception in an instance of legate::TaskException. Note that throwing exceptions from tasks "
    "is discouraged as it has severe performance implications. For example, the runtime is "
    "required to block the caller on the completion of the task. It should only be used as a "
    "last resort."};

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
                 "\" threw an exception\n\"",
                 exn_text,
                 "\"\n, but the task did not declare any exception. ",
                 EXN_EXPLANATION);
  };
  const auto report_unsupported_exception_type = [&](std::string_view exn_type,
                                                     std::string_view exn_message) {
    LEGATE_ABORT("Task \"",
                 get_task_name(),
                 "\" threw an unexpected exception (",
                 exn_type,
                 "): \"",
                 exn_message,
                 "\". ",
                 EXN_EXPLANATION);
  };

  try {
    if (!Config::get_config().use_empty_task()) {
      (*variant_impl)(ctx);
    }

    if (auto&& excn = ctx.impl()->get_exception(); excn.has_value()) {
      LEGATE_ASSERT(excn->kind() == ExceptionKind::PYTHON);
      auto msg = std::get<ReturnedPythonException>(excn->variant()).message();

      return handle_raised_exception(*std::move(excn), msg);
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
