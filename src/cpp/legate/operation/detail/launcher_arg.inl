/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/launcher_arg.h>

namespace legate::detail {

inline void AnalyzableBase::record_unbound_stores(
  SmallVector<const OutputRegionArg*>& /* args */) const
{
}

inline void AnalyzableBase::perform_invalidations() const {}

// ==========================================================================================

inline ScalarArg::ScalarArg(InternalSharedPtr<Scalar> scalar) : scalar_{std::move(scalar)} {}

// ==========================================================================================

inline RegionFieldArg::RegionFieldArg(LogicalStore* store,
                                      Legion::PrivilegeMode privilege,
                                      StoreProjection store_proj)
  : store_{store}, privilege_{privilege}, store_proj_{std::move(store_proj)}
{
}

// ==========================================================================================

inline OutputRegionArg::OutputRegionArg(LogicalStore* store,
                                        Legion::FieldSpace field_space,
                                        Legion::FieldID field_id,
                                        Legion::ProjectionID proj_id,
                                        Legion::IndexSpace color_space)
  : store_{store},
    field_space_{std::move(field_space)},
    field_id_{field_id},
    proj_id_{proj_id},
    color_space_{std::move(color_space)}
{
}

inline LogicalStore* OutputRegionArg::store() const { return store_; }

inline const Legion::FieldSpace& OutputRegionArg::field_space() const { return field_space_; }

inline Legion::FieldID OutputRegionArg::field_id() const { return field_id_; }

inline std::uint32_t OutputRegionArg::requirement_index() const { return requirement_index_; }

// ==========================================================================================

inline ScalarStoreArg::ScalarStoreArg(LogicalStore* store,
                                      Legion::Future future,
                                      std::size_t scalar_offset,
                                      bool read_only,
                                      GlobalRedopID redop)
  : store_{store},
    future_{std::move(future)},
    scalar_offset_{scalar_offset},
    read_only_{read_only},
    redop_{redop}
{
}

// ==========================================================================================

inline ReplicatedScalarStoreArg::ReplicatedScalarStoreArg(LogicalStore* store,
                                                          Legion::FutureMap future_map,
                                                          std::size_t scalar_offset,
                                                          bool read_only)
  : store_{store},
    future_map_{std::move(future_map)},
    scalar_offset_{scalar_offset},
    read_only_{read_only}
{
}

// ==========================================================================================

inline WriteOnlyScalarStoreArg::WriteOnlyScalarStoreArg(LogicalStore* store, GlobalRedopID redop)
  : store_{store}, redop_{redop}
{
}

inline void WriteOnlyScalarStoreArg::analyze(StoreAnalyzer& /*analyzer*/) const {}

}  // namespace legate::detail
