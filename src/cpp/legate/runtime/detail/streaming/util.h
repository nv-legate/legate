/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/typedefs.h>

#include <string>

namespace legate::detail {

/**
 * @brief Helper class to collect helpful info about why the streaming checker
 * failed, which operation was not, and why.
 */
class StreamingErrorContext {
 public:
  /**
   * @brief Constructor.
   *
   * @param strict_mode true if the streaming mode is STRICT. Used to disable
   *  error message collection when not needed.
   */
  explicit StreamingErrorContext(bool strict_mode);

  /**
   * @brief add helpful message to the existing context.
   *
   * @param fmt_str format string.
   * @param args printable arguments to prepend to the error message.
   */
  template <typename S, typename... Args>
  void append(const S& fmt_str, Args&&... args);

  /**
   * @return a const reference to the accumulated error string.
   */
  [[nodiscard]] const std::string& to_string() const;

 private:
  std::string context_{};
  bool enabled_{true};
};

/**
 * @return A reference to logger for streaming.
 */
[[nodiscard]] legate::Logger& log_streaming();

}  // namespace legate::detail

#include <legate/runtime/detail/streaming/util.inl>
