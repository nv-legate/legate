/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_stores/unbound_physical_store.h>

#include <utility>

namespace legate::detail {

inline UnboundPhysicalStore::UnboundPhysicalStore(
  std::int32_t dim,
  InternalSharedPtr<Type> type,
  UnboundRegionField&& unbound_field,
  InternalSharedPtr<detail::TransformStack> transform)
  : PhysicalStore{dim,
                  std::move(type),
                  GlobalRedopID{-1},
                  std::move(transform),
                  /*readable=*/false,
                  /*writable=*/false,
                  /*reducible=*/false},
    unbound_field_{std::move(unbound_field)}
{
}

inline bool UnboundPhysicalStore::valid() const { return true; }

inline bool UnboundPhysicalStore::is_partitioned() const { return unbound_field_.is_partitioned(); }

inline std::pair<Legion::OutputRegion, Legion::FieldID> UnboundPhysicalStore::get_output_field()
  const
{
  return {unbound_field_.get_output_region(), unbound_field_.get_field_id()};
}

}  // namespace legate::detail
