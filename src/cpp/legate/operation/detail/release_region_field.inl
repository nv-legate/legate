/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/release_region_field.h>

namespace legate::detail {

inline ReleaseRegionField::ReleaseRegionField(
  std::uint64_t unique_id,
  InternalSharedPtr<LogicalRegionField::PhysicalState> physical_state,
  bool unmap,
  bool unordered)
  : Operation{unique_id},
    physical_state_{std::move(physical_state)},
    unmap_{unmap},
    unordered_{unordered}
{
}

inline Operation::Kind ReleaseRegionField::kind() const { return Kind::RELEASE_REGION_FIELD; }

}  // namespace legate::detail
