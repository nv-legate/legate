/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/operation/detail/timing.h>

#include <legate/runtime/detail/runtime.h>

namespace legate::detail {

void Timing::launch()
{
  const auto get_timestamp = [&] {
    auto* const runtime = Runtime::get_runtime();
    switch (precision_) {
      case Precision::MICRO: {
        return runtime->get_legion_runtime()->get_current_time_in_microseconds(
          runtime->get_legion_context());
      }
      case Precision::NANO: {
        return runtime->get_legion_runtime()->get_current_time_in_nanoseconds(
          runtime->get_legion_context());
      }
    }
    LEGATE_UNREACHABLE();
    return Legion::Future{};
  };
  store_->set_future(get_timestamp(), 0);
}

void Timing::launch(Strategy* /*strategy*/) { launch(); }

}  // namespace legate::detail
