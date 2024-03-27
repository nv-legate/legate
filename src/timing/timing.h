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

#pragma once

#include <cstdint>
#include <memory>
#include <utility>

/**
 * @file
 * @brief Class definition legate::timing::Time
 */

namespace legate::timing {

/**
 * @ingroup util
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

  Time() = default;

 private:
  class Impl;
  explicit Time(std::shared_ptr<Impl> impl) : impl_{std::move(impl)} {}
  std::shared_ptr<Impl> impl_{};

  friend Time measure_microseconds();
  friend Time measure_nanoseconds();
};

/**
 * @ingroup util
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
 * @ingroup util
 * @brief Returns a timestamp at the resolution of nanoseconds
 *
 * The returned timestamp indicates the time at which all preceding Legate operations finish. This
 * timestamp generation is a non-blocking operation, and the blocking happens when the value wrapped
 * within the returned `Time` object is retrieved.
 *
 * @return A `Time` object
 */
[[nodiscard]] Time measure_nanoseconds();

}  // namespace legate::timing
