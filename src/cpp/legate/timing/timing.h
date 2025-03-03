/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/shared_ptr.h>

#include <cstdint>

/**
 * @file
 * @brief Class definition legate::timing::Time
 */

namespace legate::timing {

/**
 * @addtogroup util
 * @{
 */

/**
 * @brief Deferred timestamp class
 */
class Time {
 public:
  /**
   * @brief Returns the timestamp value in this `Time` object
   *
   * Blocks on all Legate operations preceding the call that generated this `Time` object.
   *
   * @return A timestamp value
   */
  [[nodiscard]] std::int64_t value() const;

  Time()                           = LEGATE_DEFAULT_WHEN_CYTHON;
  Time(const Time&)                = default;
  Time& operator=(const Time&)     = default;
  Time(Time&&) noexcept            = default;
  Time& operator=(Time&&) noexcept = default;
  ~Time();

 private:
  class Impl;

  explicit Time(SharedPtr<Impl> impl);

  SharedPtr<Impl> impl_{};

  friend Time measure_microseconds();
  friend Time measure_nanoseconds();
};

/**
 * @brief Returns a timestamp at the resolution of microseconds
 *
 * The returned timestamp indicates the time at which all preceding Legate operations finish. This
 * timestamp generation is a non-blocking operation, and the blocking happens when the value wrapped
 * within the returned `Time` object is retrieved.
 *
 * @return A `Time` object
 */
[[nodiscard]] Time measure_microseconds();

/**
 * @brief Returns a timestamp at the resolution of nanoseconds
 *
 * The returned timestamp indicates the time at which all preceding Legate operations finish. This
 * timestamp generation is a non-blocking operation, and the blocking happens when the value wrapped
 * within the returned `Time` object is retrieved.
 *
 * @return A `Time` object
 */
[[nodiscard]] Time measure_nanoseconds();

/** @} */

}  // namespace legate::timing
