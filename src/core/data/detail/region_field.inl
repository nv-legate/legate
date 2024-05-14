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

#include "core/data/detail/region_field.h"

namespace legate::detail {

inline std::int32_t RegionField::dim() const { return dim_; }

inline void RegionField::set_logical_region(const Legion::LogicalRegion& lr) { lr_ = lr; }

inline bool RegionField::is_readable() const { return readable_; }

inline bool RegionField::is_writable() const { return writable_; }

inline bool RegionField::is_reducible() const { return reducible_; }

inline const Legion::PhysicalRegion& RegionField::get_physical_region() const
{
  LegateAssert(pr_.has_value());
  return *pr_;  // NOLINT(bugprone-unchecked-optional-access)
}

inline Legion::FieldID RegionField::get_field_id() const { return fid_; }

}  // namespace legate::detail
