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

#include "core/runtime/detail/runtime.h"

#include "legion.h"

#include <optional>

namespace legate::timing {

class Time::Impl {
 public:
  explicit Impl(Legion::Future future)
    : future_{std::make_unique<Legion::Future>(std::move(future))}
  {
  }

  ~Impl()
  {
    if (!detail::Runtime::get_runtime()->initialized()) {
      // Leak the Future handle if the runtime has already shut down, as there's no hope that
      // this would be collected by the Legion runtime
      static_cast<void>(future_.release());  // NOLINT(bugprone-unused-return-value)
    }
  }

  [[nodiscard]] std::int64_t value()
  {
    if (!value_) {
      value_ = future_->get_result<std::int64_t>();
    }
    return *value_;
  }

 private:
  std::unique_ptr<Legion::Future> future_{};
  std::optional<std::int64_t> value_{std::nullopt};
};

std::int64_t Time::value() const { return impl_->value(); }

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
