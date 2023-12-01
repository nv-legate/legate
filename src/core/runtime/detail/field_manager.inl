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

#include "core/runtime/detail/field_manager.h"

#include <tuple>

namespace legate::detail {

inline FreeFieldInfo::FreeFieldInfo(Legion::LogicalRegion region_,
                                    Legion::FieldID field_id_,
                                    Legion::Future can_dealloc_,
                                    void* attachment_)
  : region{std::move(region_)},
    field_id{std::move(field_id_)},
    can_dealloc{std::move(can_dealloc_)},
    attachment{attachment_}
{
}

// ==========================================================================================

inline MatchItem::MatchItem(Legion::RegionTreeID tid_, Legion::FieldID fid_) : tid{tid_}, fid{fid_}
{
}

inline bool operator<(const MatchItem& l, const MatchItem& r)
{
  return std::tie(l.tid, l.fid) < std::tie(r.tid, r.fid);
}

}  // namespace legate::detail
