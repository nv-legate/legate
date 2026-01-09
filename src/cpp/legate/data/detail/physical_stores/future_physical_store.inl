/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_stores/future_physical_store.h>

#include <utility>

namespace legate::detail {

inline FuturePhysicalStore::FuturePhysicalStore(std::int32_t dim,
                                                InternalSharedPtr<Type> type,
                                                GlobalRedopID redop_id,
                                                FutureWrapper future,
                                                InternalSharedPtr<detail::TransformStack> transform)
  : PhysicalStore{dim,
                  std::move(type),
                  redop_id,
                  std::move(transform),
                  future.valid(),
                  !future.is_read_only(),
                  !future.is_read_only()},
    future_{std::move(future)}
{
}

inline PhysicalStore::Kind FuturePhysicalStore::kind() const { return Kind::FUTURE; }

inline bool FuturePhysicalStore::valid() const { return true; }

inline bool FuturePhysicalStore::is_partitioned() const { return false; }

inline const Legion::Future& FuturePhysicalStore::get_future() const
{
  return future_.get_future();
}

inline const Legion::UntypedDeferredValue& FuturePhysicalStore::get_buffer() const
{
  return future_.get_buffer();
}

inline bool FuturePhysicalStore::is_read_only_future() const { return future_.is_read_only(); }

inline std::size_t FuturePhysicalStore::get_field_offset() const { return future_.field_offset(); }

inline const void* FuturePhysicalStore::get_untyped_pointer_from_future() const
{
  return future_.get_untyped_pointer_from_future();
}

}  // namespace legate::detail
