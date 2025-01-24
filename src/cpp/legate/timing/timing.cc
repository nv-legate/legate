/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/timing/timing.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/physical_store.h>
#include <legate/operation/detail/timing.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/traced_exception.h>

#include <legion.h>

#include <optional>
#include <stdexcept>

namespace legate::timing {

class Time::Impl {
 public:
  explicit Impl(InternalSharedPtr<detail::LogicalStore> store) : store_{std::move(store)} {}

  [[nodiscard]] std::int64_t value()
  {
    if (!value_) {
      value_ = legate::PhysicalStore{store_->get_physical_store(false)}.scalar<std::int64_t>();
      store_.reset();
    }
    return *value_;
  }

 private:
  InternalSharedPtr<detail::LogicalStore> store_{};
  std::optional<std::int64_t> value_{std::nullopt};
};

std::int64_t Time::value() const
{
  if (!impl_) {
    throw legate::detail::TracedException<std::invalid_argument>{"Invalid time object"};
  }
  return impl_->value();
}

Time measure_microseconds()
{
  return Time{make_shared<Time::Impl>(
    detail::Runtime::get_runtime()->get_timestamp(detail::Timing::Precision::MICRO))};
}

Time measure_nanoseconds()
{
  return Time{make_shared<Time::Impl>(
    detail::Runtime::get_runtime()->get_timestamp(detail::Timing::Precision::NANO))};
}

}  // namespace legate::timing
