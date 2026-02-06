/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/region_field.h>

namespace legate::detail {

inline std::int32_t RegionField::dim() const { return dim_; }

inline void RegionField::set_logical_region(const Legion::LogicalRegion& lr) { lr_ = lr; }

inline bool RegionField::is_readable() const { return readable_; }

inline bool RegionField::is_writable() const { return writable_; }

inline bool RegionField::is_reducible() const { return reducible_; }

inline bool RegionField::is_partitioned() const { return partitioned_; }

inline const Legion::PhysicalRegion& RegionField::get_physical_region() const
{
  LEGATE_ASSERT(pr_.has_value());
  return *pr_;  // NOLINT(bugprone-unchecked-optional-access)
}

inline Legion::FieldID RegionField::get_field_id() const { return fid_; }

inline Legion::LogicalRegion RegionField::get_logical_region() const { return lr_; }

}  // namespace legate::detail
