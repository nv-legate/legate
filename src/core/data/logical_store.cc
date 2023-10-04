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

#include <numeric>

#include "core/data/detail/logical_store.h"
#include "core/data/logical_store.h"
#include "core/data/store.h"
#include "core/partitioning/partition.h"
#include "core/type/detail/type_info.h"
#include "core/type/type_traits.h"

namespace legate {

extern Logger log_legate;

LogicalStore::LogicalStore(std::shared_ptr<detail::LogicalStore>&& impl) : impl_(std::move(impl)) {}

int32_t LogicalStore::dim() const { return impl_->dim(); }

Type LogicalStore::type() const { return Type(impl_->type()); }

const Shape& LogicalStore::extents() const { return impl_->extents(); }

size_t LogicalStore::volume() const { return impl_->volume(); }

bool LogicalStore::unbound() const { return impl_->unbound(); }

bool LogicalStore::transformed() const { return impl_->transformed(); }

LogicalStore LogicalStore::promote(int32_t extra_dim, size_t dim_size) const
{
  return LogicalStore(impl_->promote(extra_dim, dim_size));
}

LogicalStore LogicalStore::project(int32_t dim, int64_t index) const
{
  return LogicalStore(impl_->project(dim, index));
}

LogicalStorePartition LogicalStore::partition_by_tiling(std::vector<size_t> tile_shape) const
{
  return LogicalStorePartition(impl_->partition_by_tiling(Shape(std::move(tile_shape))));
}

LogicalStore LogicalStore::slice(int32_t dim, Slice sl) const
{
  return LogicalStore(impl_->slice(dim, sl));
}

LogicalStore LogicalStore::transpose(std::vector<int32_t>&& axes) const
{
  return LogicalStore(impl_->transpose(std::move(axes)));
}

LogicalStore LogicalStore::delinearize(int32_t dim, std::vector<int64_t>&& sizes) const
{
  return LogicalStore(impl_->delinearize(dim, std::move(sizes)));
}

Store LogicalStore::get_physical_store() const { return Store(impl_->get_physical_store()); }

void LogicalStore::detach() { impl_->detach(); }

LogicalStorePartition::LogicalStorePartition(std::shared_ptr<detail::LogicalStorePartition>&& impl)
  : impl_(std::move(impl))
{
}

LogicalStore LogicalStorePartition::store() const { return LogicalStore(impl_->store()); }

const Shape& LogicalStorePartition::color_shape() const
{
  return impl_->partition()->color_shape();
}

}  // namespace legate
