/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "timing/timing.h"

#include "legion.h"

#include <optional>

namespace legate::timing {

class Time::Impl {
 public:
  explicit Impl(Legion::Future future) : future_{std::move(future)} {}

  [[nodiscard]] int64_t value()
  {
    if (!value_) {
      value_ = future_.get_result<int64_t>();
    }
    return *value_;
  }

 private:
  Legion::Future future_{};
  std::optional<int64_t> value_{std::nullopt};
};

int64_t Time::value() const { return impl_->value(); }

Time measure_microseconds()
{
  auto runtime = Legion::Runtime::get_runtime();
  auto context = Legion::Runtime::get_context();

  auto future = runtime->get_current_time_in_microseconds(context);

  return Time{std::make_shared<Time::Impl>(std::move(future))};
}

Time measure_nanoseconds()
{
  auto runtime = Legion::Runtime::get_runtime();
  auto context = Legion::Runtime::get_context();

  auto future = runtime->get_current_time_in_nanoseconds(context);

  return Time{std::make_shared<Time::Impl>(std::move(future))};
}

}  // namespace legate::timing
