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
    LEGATE_ABORT("Unhandled precision ", traits::detail::to_underlying(precision_));
    return Legion::Future{};
  };
  store_->set_future(get_timestamp(), 0);
}

void Timing::launch(Strategy* /*strategy*/) { launch(); }

}  // namespace legate::detail
