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

#pragma once

#include <cstdint>
#include <optional>

/**
 * @file
 * @brief A simple slice class that has the same semantics as Python's
 */

namespace legate {

/**
 * @ingroup data
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

}  // namespace legate

#include "core/data/slice.inl"
