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

#include "core/data/logical_store.h"

#include "core/data/detail/logical_store.h"
#include "core/data/physical_store.h"

namespace legate {

int32_t LogicalStore::dim() const { return impl_->dim(); }

Type LogicalStore::type() const { return Type{impl_->type()}; }

const Shape& LogicalStore::extents() const { return impl_->extents(); }

size_t LogicalStore::volume() const { return impl_->volume(); }

bool LogicalStore::unbound() const { return impl_->unbound(); }

bool LogicalStore::transformed() const { return impl_->transformed(); }

bool LogicalStore::has_scalar_storage() const { return impl_->has_scalar_storage(); }

bool LogicalStore::overlaps(const LogicalStore& other) const
{
  return impl_->overlaps(other.impl_);
}

LogicalStore LogicalStore::promote(int32_t extra_dim, size_t dim_size) const
{
  return LogicalStore{impl_->promote(extra_dim, dim_size)};
}

LogicalStore LogicalStore::project(int32_t dim, int64_t index) const
{
  return LogicalStore{impl_->project(dim, index)};
}

LogicalStorePartition LogicalStore::partition_by_tiling(std::vector<size_t> tile_shape) const
{
  return LogicalStorePartition{impl_->partition_by_tiling(Shape{std::move(tile_shape)})};
}

LogicalStore LogicalStore::slice(int32_t dim, Slice sl) const
{
  return LogicalStore{impl_->slice(dim, sl)};
}

LogicalStore LogicalStore::transpose(std::vector<int32_t>&& axes) const
{
  return LogicalStore{impl_->transpose(std::move(axes))};
}

LogicalStore LogicalStore::delinearize(int32_t dim, std::vector<uint64_t> sizes) const
{
  return LogicalStore{impl_->delinearize(dim, std::move(sizes))};
}

PhysicalStore LogicalStore::get_physical_store() const
{
  return PhysicalStore{impl_->get_physical_store()};
}

std::string LogicalStore::to_string() const { return impl_->to_string(); }

void LogicalStore::detach() { impl_->detach(); }

LogicalStore LogicalStorePartition::store() const { return LogicalStore{impl_->store()}; }

const Shape& LogicalStorePartition::color_shape() const { return impl_->color_shape(); }

LogicalStore LogicalStorePartition::get_child_store(const Shape& color) const
{
  return LogicalStore{impl_->get_child_store(color)};
}

}  // namespace legate
