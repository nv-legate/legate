/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>

namespace legate::experimental {

class Trace {
 public:
  explicit Trace(std::uint32_t trace_id);
  ~Trace();

  static void begin_trace(std::uint32_t trace_id);
  static void end_trace(std::uint32_t trace_id);

  Trace(const Trace&)            = delete;
  Trace& operator=(const Trace&) = delete;
  Trace(Trace&&)                 = delete;
  Trace& operator=(Trace&&)      = delete;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_{};
};

}  // namespace legate::experimental
