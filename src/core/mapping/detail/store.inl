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

#include "core/mapping/detail/store.h"

namespace legate::mapping::detail {

inline RegionField::RegionField(const Legion::RegionRequirement* req,
                                int32_t dim,
                                uint32_t idx,
                                Legion::FieldID fid)
  : req_{req}, dim_{dim}, idx_{idx}, fid_{fid}
{
}

inline RegionField::Id RegionField::unique_id() const { return {unbound(), index(), field_id()}; }

inline int32_t RegionField::dim() const { return dim_; }

inline uint32_t RegionField::index() const { return idx_; }

inline Legion::FieldID RegionField::field_id() const { return fid_; }

inline bool RegionField::unbound() const { return dim() < 0; }

inline const Legion::RegionRequirement* RegionField::get_requirement() const { return req_; }

// ==========================================================================================

// Silence pass-by-value since Legion::Domain is POD, and the move ctor just does the copy
// anyways. Unfortunately there is no way to check this programatically (e.g. via a
// static_assert).
inline FutureWrapper::FutureWrapper(uint32_t idx,
                                    const Legion::Domain& domain  // NOLINT(modernize-pass-by-value)
                                    )
  : idx_{idx}, domain_{domain}
{
}

inline int32_t FutureWrapper::dim() const { return domain_.dim; }

inline uint32_t FutureWrapper::index() const { return idx_; }

inline Legion::Domain FutureWrapper::domain() const { return domain_; }

// ==========================================================================================

inline bool Store::is_future() const { return is_future_; }

inline bool Store::unbound() const { return is_unbound_store_; }

inline int32_t Store::dim() const { return dim_; }

inline InternalSharedPtr<legate::detail::Type> Store::type() const { return type_; }

inline bool Store::is_reduction() const { return redop() > 0; }

inline int32_t Store::redop() const { return redop_id_; }

inline RegionField::Id Store::unique_region_field_id() const { return region_field().unique_id(); }

inline uint32_t Store::requirement_index() const { return region_field().index(); }

inline uint32_t Store::future_index() const { return future().index(); }

}  // namespace legate::mapping::detail
