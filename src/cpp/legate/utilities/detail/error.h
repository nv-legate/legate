/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/span.h>

#include <cpptrace/basic.hpp>

#include <string>
#include <vector>

namespace legate::detail {

class ErrorDescription {
 public:
  explicit ErrorDescription(std::string single_line, cpptrace::stacktrace stack_trace = {});
  ErrorDescription(std::vector<std::string> lines, cpptrace::stacktrace stack_trace);

  std::vector<std::string> message_lines{};
  cpptrace::stacktrace trace{};
};

[[nodiscard]] std::string make_error_message(Span<const ErrorDescription> errs);

}  // namespace legate::detail
