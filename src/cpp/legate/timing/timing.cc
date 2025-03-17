/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/timing/timing.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/physical_store.h>
#include <legate/mapping/mapping.h>
#include <legate/operation/detail/timing.h>
#include <legate/runtime/detail/runtime.h>

#include <optional>
#include <utility>

namespace legate::timing {

class Time::Impl {
 public:
  explicit Impl(InternalSharedPtr<detail::LogicalStore> store) : store_{std::move(store)} {}

  [[nodiscard]] std::int64_t value()
  {
    if (!value_) {
      value_ =
        legate::PhysicalStore{store_->get_physical_store(legate::mapping::StoreTarget::SYSMEM,
                                                         /* ignore_future_mutability */ false)}
          .scalar<std::int64_t>();
      store_.reset();
    }
    return *value_;
  }

 private:
  InternalSharedPtr<detail::LogicalStore> store_{};
  std::optional<std::int64_t> value_{std::nullopt};
};

Time::Time(SharedPtr<Impl> impl) : impl_{std::move(impl)} {}

std::int64_t Time::value() const { return impl_->value(); }

Time::~Time() = default;

// ==========================================================================================

Time measure_microseconds()
{
  return Time{legate::make_shared<Time::Impl>(
    detail::Runtime::get_runtime()->get_timestamp(detail::Timing::Precision::MICRO))};
}

Time measure_nanoseconds()
{
  return Time{legate::make_shared<Time::Impl>(
    detail::Runtime::get_runtime()->get_timestamp(detail::Timing::Precision::NANO))};
}

}  // namespace legate::timing
