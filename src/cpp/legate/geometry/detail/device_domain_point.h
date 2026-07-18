/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <cstdint>

namespace legate::detail {

/**
 * @brief A trivially-copyable, device-usable stand-in for `legate::DomainPoint`.
 *
 * `legate::DomainPoint` stores its coordinates in a `SmallVector`, which is neither trivially
 * copyable nor usable in device code, so it cannot be passed by value to a CUDA kernel. This
 * plain struct mirrors the coordinate storage and can be memcpy'd into kernel parameter space.
 * Host code converts a `DomainPoint` into a `DeviceDomainPoint` right before launching a kernel
 * (see `to_kernel_domain_point()` in copy_request.cc).
 */
class DeviceDomainPoint {
 public:
  [[nodiscard]] LEGATE_HOST_DEVICE std::int64_t& operator[](std::int32_t index) noexcept;
  [[nodiscard]] LEGATE_HOST_DEVICE const std::int64_t& operator[](
    std::int32_t index) const noexcept;

  std::uint32_t dim{};
  std::int64_t coords[LEGATE_MAX_DIM]{};
};

}  // namespace legate::detail

#include <legate/geometry/detail/device_domain_point.inl>
