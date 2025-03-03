/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/timing.h>

#include <legate/utilities/abort.h>
#include <legate/utilities/detail/type_traits.h>

namespace legate::detail {

void Timing::launch()
{
  const auto get_timestamp = [&] {
    switch (precision_) {
      // Don't use legate::Runtime::get_legion_context() here, since timing calls are allowed inside
      // non-top-level tasks, and get_legion_context returns the context associated with the
      // top-level task.
      case Precision::MICRO: {
        return Legion::Runtime::get_runtime()->get_current_time_in_microseconds(
          Legion::Runtime::get_context());
      }
      case Precision::NANO: {
        return Legion::Runtime::get_runtime()->get_current_time_in_nanoseconds(
          Legion::Runtime::get_context());
      }
    }
    LEGATE_ABORT("Unhandled precision ", to_underlying(precision_));
    return Legion::Future{};
  };
  store_->set_future(get_timestamp(), 0);
}

void Timing::launch(Strategy* /*strategy*/) { launch(); }

}  // namespace legate::detail
