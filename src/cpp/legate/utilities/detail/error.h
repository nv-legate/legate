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
