/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/partition/opaque.h>

namespace legate::detail {

inline Opaque::Kind Opaque::kind() const { return Kind::OPAQUE; }

inline bool Opaque::is_convertible() const { return false; }

inline bool Opaque::is_invertible() const { return false; }

inline bool Opaque::is_complete_for(const detail::Storage& /*storage*/) const
{
  // We use Opaque partitions only for unbound stores, so they are always complete
  return true;
}

inline bool Opaque::has_launch_domain() const { return true; }

inline Domain Opaque::launch_domain() const { return color_domain_; }

inline Span<const std::uint64_t> Opaque::color_shape() const { return color_shape_; }

}  // namespace legate::detail
