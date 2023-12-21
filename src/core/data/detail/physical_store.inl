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

#include "core/data/detail/physical_store.h"

#include <utility>

namespace legate::detail {

inline int32_t RegionField::dim() const { return dim_; }

inline void RegionField::set_logical_region(const Legion::LogicalRegion& lr) { lr_ = lr; }

inline bool RegionField::is_readable() const { return readable_; }

inline bool RegionField::is_writable() const { return writable_; }

inline bool RegionField::is_reducible() const { return reducible_; }

inline Legion::PhysicalRegion RegionField::get_physical_region() const { return *pr_; }

inline Legion::FieldID RegionField::get_field_id() const { return fid_; }

// ==========================================================================================

inline UnboundRegionField::UnboundRegionField(const Legion::OutputRegion& out, Legion::FieldID fid)
  : num_elements_{Legion::UntypedDeferredValue(sizeof(size_t),
                                               find_memory_kind_for_executing_processor())},
    out_{out},
    fid_{fid}
{
}

inline UnboundRegionField::UnboundRegionField(UnboundRegionField&& other) noexcept
  : bound_{std::exchange(other.bound_, false)},
    num_elements_{std::exchange(other.num_elements_, Legion::UntypedDeferredValue())},
    out_{std::exchange(other.out_, Legion::OutputRegion())},
    fid_{std::exchange(other.fid_, -1)}
{
}

inline bool UnboundRegionField::bound() const { return bound_; }

inline void UnboundRegionField::set_bound(bool bound) { bound_ = bound; }

inline Legion::OutputRegion UnboundRegionField::get_output_region() const { return out_; }

inline Legion::FieldID UnboundRegionField::get_field_id() const { return fid_; }

// ==========================================================================================

inline int32_t FutureWrapper::dim() const { return domain_.dim; }

inline Domain FutureWrapper::domain() const { return domain_; }

inline bool FutureWrapper::valid() const { return future_ != nullptr && future_->valid(); }

inline bool FutureWrapper::is_read_only() const { return read_only_; }

inline Legion::UntypedDeferredValue FutureWrapper::get_buffer() const { return buffer_; }

// ==========================================================================================

inline PhysicalStore::PhysicalStore(int32_t dim,
                                    InternalSharedPtr<Type> type,
                                    int32_t redop_id,
                                    FutureWrapper future,
                                    InternalSharedPtr<detail::TransformStack>&& transform)
  : is_future_{true},
    dim_{dim},
    type_{std::move(type)},
    redop_id_{redop_id},
    future_{std::move(future)},
    transform_{std::move(transform)},
    readable_{future_.valid()},
    writable_{!future_.is_read_only()}
{
}

inline PhysicalStore::PhysicalStore(int32_t dim,
                                    InternalSharedPtr<Type> type,
                                    int32_t redop_id,
                                    RegionField&& region_field,
                                    InternalSharedPtr<detail::TransformStack>&& transform)
  : dim_{dim},
    type_{std::move(type)},
    redop_id_{redop_id},
    region_field_{std::move(region_field)},
    transform_{std::move(transform)},
    readable_{region_field_.is_readable()},
    writable_{region_field_.is_writable()},
    reducible_{region_field_.is_reducible()}
{
}

inline PhysicalStore::PhysicalStore(int32_t dim,
                                    InternalSharedPtr<Type> type,
                                    UnboundRegionField&& unbound_field,
                                    InternalSharedPtr<detail::TransformStack>&& transform)
  : is_unbound_store_{true},
    dim_{dim},
    type_{std::move(type)},
    unbound_field_{std::move(unbound_field)},
    transform_{std::move(transform)}
{
}

inline PhysicalStore::PhysicalStore(int32_t dim,
                                    InternalSharedPtr<Type> type,
                                    int32_t redop_id,
                                    FutureWrapper future,
                                    const InternalSharedPtr<detail::TransformStack>& transform)
  : is_future_{true},
    dim_{dim},
    type_{std::move(type)},
    redop_id_{redop_id},
    future_{std::move(future)},
    transform_{transform},
    readable_{true}
{
}

inline PhysicalStore::PhysicalStore(int32_t dim,
                                    InternalSharedPtr<Type> type,
                                    int32_t redop_id,
                                    RegionField&& region_field,
                                    const InternalSharedPtr<detail::TransformStack>& transform)
  : dim_{dim},
    type_{std::move(type)},
    redop_id_{redop_id},
    region_field_{std::move(region_field)},
    transform_{transform},
    readable_{region_field_.is_readable()},
    writable_{region_field_.is_writable()},
    reducible_{region_field_.is_reducible()}
{
}

inline int32_t PhysicalStore::dim() const { return dim_; }

inline const InternalSharedPtr<Type>& PhysicalStore::type() const { return type_; }

inline bool PhysicalStore::is_readable() const { return readable_; }

inline bool PhysicalStore::is_writable() const { return writable_; }

inline bool PhysicalStore::is_reducible() const { return reducible_; }

inline bool PhysicalStore::is_future() const { return is_future_; }

inline bool PhysicalStore::is_unbound_store() const { return is_unbound_store_; }

inline ReturnValue PhysicalStore::pack() const { return future_.pack(); }

inline ReturnValue PhysicalStore::pack_weight() const { return unbound_field_.pack_weight(); }

inline int32_t PhysicalStore::get_redop_id() const { return redop_id_; }

}  // namespace legate::detail
