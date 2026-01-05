/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/detail/field_manager.h>

namespace legate::detail {

inline FreeFieldInfo::FreeFieldInfo(InternalSharedPtr<Shape> shape_,
                                    std::uint32_t field_size_,
                                    Legion::LogicalRegion region_,
                                    Legion::FieldID field_id_,
                                    InternalSharedPtr<LogicalRegionField::PhysicalState> state_)
  : shape{std::move(shape_)},
    field_size{field_size_},
    region{std::move(region_)},
    field_id{std::move(field_id_)},
    state{std::move(state_)}
{
}

// ==========================================================================================

inline MatchItem::MatchItem(Legion::RegionTreeID tid_, Legion::FieldID fid_) : tid{tid_}, fid{fid_}
{
}

inline bool MatchItem::operator==(const MatchItem& rhs) const
{
  return tid == rhs.tid && fid == rhs.fid;
}

inline std::size_t MatchItem::hash() const noexcept { return hash_all(tid, fid); }

}  // namespace legate::detail
