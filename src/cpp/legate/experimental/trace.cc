/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/experimental/trace.h>

#include <legate/runtime/detail/runtime.h>

namespace legate::experimental {

class Trace::Impl {
 public:
  explicit Impl(std::uint32_t trace_id) : trace_id_{trace_id} { Trace::begin_trace(trace_id_); }

  ~Impl() { Trace::end_trace(trace_id_); }

 private:
  std::uint32_t trace_id_{};
};

Trace::Trace(std::uint32_t trace_id) : impl_{std::make_unique<Trace::Impl>(trace_id)} {}

Trace::~Trace() = default;

/*static*/ void Trace::begin_trace(std::uint32_t trace_id)
{
  detail::Runtime::get_runtime().begin_trace(trace_id);
}

/*static*/ void Trace::end_trace(std::uint32_t trace_id)
{
  detail::Runtime::get_runtime().end_trace(trace_id);
}

}  // namespace legate::experimental
