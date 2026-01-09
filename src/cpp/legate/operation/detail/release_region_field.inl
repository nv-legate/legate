/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/release_region_field.h>

namespace legate::detail {

inline ReleaseRegionField::ReleaseRegionField(
  std::uint64_t unique_id,
  InternalSharedPtr<LogicalRegionField::PhysicalState> physical_state,
  bool unordered)
  : Operation{unique_id}, physical_state_{std::move(physical_state)}, unordered_{unordered}
{
}

inline Operation::Kind ReleaseRegionField::kind() const { return Kind::RELEASE_REGION_FIELD; }

inline bool ReleaseRegionField::needs_flush() const { return false; }

inline bool ReleaseRegionField::needs_partitioning() const { return false; }

}  // namespace legate::detail
