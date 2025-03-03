/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>

#include <cstdint>
#include <optional>

/**
 * @file
 * @brief A simple slice class that has the same semantics as Python's
 */

namespace legate {

/**
 * @addtogroup data
 * @{
 */

/**
 * @brief A slice descriptor
 *
 * `Slice` behaves similarly to how the slice in Python does, and has different semantics
 * from `std::slice`.
 */
class Slice {
 public:
  static constexpr std::nullopt_t OPEN = std::nullopt;

  /**
   * @brief Constructs a `Slice`
   *
   * @param _start The optional begin index of the slice, or `Slice::OPEN` if the start of the
   * slice is unbounded.
   * @param _stop The optional stop index of the slice, or `Slice::OPEN` if the end of the
   * slice if unbounded.
   *
   * If provided (and not `Slice::OPEN`), `_start` must compare less than or equal to
   * `_stop`. Similarly, if provided (and not `Slice::OPEN`), `_stop` must compare greater than
   * or equal to`_start`. Put simply, unless one or both of the ends are unbounded, `[_start,
   * _stop]` must form a valid (possibly empty) interval.
   */
  // NOLINTNEXTLINE(google-explicit-constructor)
  Slice(std::optional<std::int64_t> _start = OPEN, std::optional<std::int64_t> _stop = OPEN);

  std::optional<std::int64_t> start{OPEN}; /**< The start index of the slice */
  std::optional<std::int64_t> stop{OPEN};  /**< The end index of the slice */
};

/** @} */

}  // namespace legate

#include <legate/data/slice.inl>
