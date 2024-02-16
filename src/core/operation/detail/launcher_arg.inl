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

#include "core/operation/detail/launcher_arg.h"

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

inline ScalarArg::ScalarArg(Scalar&& scalar) : scalar_{std::move(scalar)} {}

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

inline FutureStoreArg::FutureStoreArg(LogicalStore* store,
                                      bool read_only,
                                      bool has_storage,
                                      Legion::ReductionOpID redop)
  : store_{store}, read_only_{read_only}, has_storage_{has_storage}, redop_{redop}
{
}

// ==========================================================================================

inline BaseArrayArg::BaseArrayArg(std::unique_ptr<Analyzable> data,
                                  std::unique_ptr<Analyzable> null_mask)
  : data_{std::move(data)}, null_mask_{std::move(null_mask)}
{
}

inline BaseArrayArg::BaseArrayArg(std::unique_ptr<Analyzable> data)
  : BaseArrayArg{std::move(data), {}}
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
                                      std::unique_ptr<Analyzable> null_mask,
                                      std::vector<std::unique_ptr<Analyzable>>&& fields)
  : type_{std::move(type)}, null_mask_{std::move(null_mask)}, fields_{std::move(fields)}
{
}

}  // namespace legate::detail
