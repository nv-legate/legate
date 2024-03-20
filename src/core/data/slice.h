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
 * legate::Slice behaves similarly to how the slice in Python does, and has different semantics
 * from std::slice.
 */
class Slice {
 public:
  static constexpr std::nullopt_t OPEN = std::nullopt;

  // NOLINTNEXTLINE(google-explicit-constructor)
  Slice(std::optional<std::int64_t> _start = OPEN, std::optional<std::int64_t> _stop = OPEN);

  std::optional<std::int64_t> start{OPEN};
  std::optional<std::int64_t> stop{OPEN};
};

}  // namespace legate

#include "core/data/slice.inl"
