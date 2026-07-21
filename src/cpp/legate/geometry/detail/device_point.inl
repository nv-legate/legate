/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace legate::detail {

inline LEGATE_HOST_DEVICE std::int64_t& DevicePoint::operator[](std::int32_t index) noexcept
{
  return coords[index];
}

inline LEGATE_HOST_DEVICE const std::int64_t& DevicePoint::operator[](
  std::int32_t index) const noexcept
{
  return coords[index];
}

}  // namespace legate::detail
