/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/launcher_arg.h>

namespace legate::detail {

inline std::optional<Legion::ProjectionID> Analyzable::get_key_proj_id() const
{
  return std::nullopt;
}

inline void Analyzable::record_unbound_stores(std::vector<const OutputRegionArg*>& /* args */) const
{
}

inline void Analyzable::perform_invalidations() const {}

// ==========================================================================================

inline ScalarArg::ScalarArg(InternalSharedPtr<Scalar> scalar) : scalar_{std::move(scalar)} {}

// ==========================================================================================

inline RegionFieldArg::RegionFieldArg(LogicalStore* store,
                                      Legion::PrivilegeMode privilege,
                                      std::unique_ptr<StoreProjection> store_proj)
  : store_{store}, privilege_{privilege}, store_proj_{std::move(store_proj)}
{
}

// ==========================================================================================

inline OutputRegionArg::OutputRegionArg(LogicalStore* store,
                                        Legion::FieldSpace field_space,
                                        Legion::FieldID field_id)
  : store_{store}, field_space_{std::move(field_space)}, field_id_{field_id}
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

inline void WriteOnlyScalarStoreArg::analyze(StoreAnalyzer& /*analyzer*/) {}

// ==========================================================================================

inline BaseArrayArg::BaseArrayArg(std::unique_ptr<Analyzable> data,
                                  std::optional<std::unique_ptr<Analyzable>> null_mask)
  : data_{std::move(data)}, null_mask_{std::move(null_mask)}
{
}

inline BaseArrayArg::BaseArrayArg(std::unique_ptr<Analyzable> data)
  : BaseArrayArg{std::move(data), std::nullopt}
{
}

// ==========================================================================================

inline ListArrayArg::ListArrayArg(InternalSharedPtr<Type> type,
                                  std::unique_ptr<Analyzable> descriptor,
                                  std::unique_ptr<Analyzable> vardata)
  : type_{std::move(type)}, descriptor_{std::move(descriptor)}, vardata_{std::move(vardata)}
{
}

// ==========================================================================================

inline StructArrayArg::StructArrayArg(InternalSharedPtr<Type> type,
                                      std::optional<std::unique_ptr<Analyzable>> null_mask,
                                      std::vector<std::unique_ptr<Analyzable>>&& fields)
  : type_{std::move(type)}, null_mask_{std::move(null_mask)}, fields_{std::move(fields)}
{
}

}  // namespace legate::detail
