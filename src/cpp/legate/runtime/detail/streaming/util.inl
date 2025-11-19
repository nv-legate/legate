/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/detail/streaming/util.h>

#include <fmt/format.h>

namespace legate::detail {

template <typename S, typename... Args>
inline void StreamingErrorContext::append(const S& fmt_str, Args&&... args)
{
  if (enabled_) {
    fmt::format_to(std::back_inserter(context_), fmt_str, std::forward<Args>(args)...);
    context_.push_back('\n');
  }
}

inline const std::string& StreamingErrorContext::to_string() const { return context_; }

}  // namespace legate::detail
