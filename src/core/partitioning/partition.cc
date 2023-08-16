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

#include <sstream>

#include "core/data/detail/logical_store.h"
#include "core/operation/detail/task.h"
#include "core/partitioning/detail/constraint.h"
#include "core/partitioning/partition.h"
#include "core/runtime/detail/partition_manager.h"
#include "core/runtime/detail/runtime.h"
#include "core/runtime/library.h"
#include "core/type/detail/type_info.h"

namespace legate {

NoPartition::NoPartition() : Partition() {}

bool NoPartition::is_complete_for(const detail::Storage* storage) const { return true; }

bool NoPartition::is_disjoint_for(const Domain* launch_domain) const
{
  return nullptr == launch_domain || launch_domain->get_volume() == 1;
}

bool NoPartition::satisfies_restrictions(const Restrictions& restrictions) const { return true; }

Legion::LogicalPartition NoPartition::construct(Legion::LogicalRegion region, bool complete) const
{
  return Legion::LogicalPartition::NO_PART;
}

bool NoPartition::has_launch_domain() const { return false; }

Legion::Domain NoPartition::launch_domain() const
{
  assert(false);
  return Legion::Domain();
}

std::unique_ptr<Partition> NoPartition::clone() const { return create_no_partition(); }

std::string NoPartition::to_string() const { return "NoPartition"; }

Tiling::Tiling(Shape&& tile_shape, Shape&& color_shape, tuple<int64_t>&& offsets)
  : Partition(),
    tile_shape_(std::move(tile_shape)),
    color_shape_(std::move(color_shape)),
    offsets_(std::move(offsets))
{
  if (offsets_.empty()) offsets_ = tuple<int64_t>(tile_shape_.size(), 0);
  assert(tile_shape_.size() == color_shape_.size());
  assert(tile_shape_.size() == offsets_.size());
}

bool Tiling::operator==(const Tiling& other) const
{
  return tile_shape_ == other.tile_shape_ && color_shape_ == other.color_shape_ &&
         offsets_ == other.offsets_;
}

bool Tiling::operator<(const Tiling& other) const
{
  if (tile_shape_ < other.tile_shape_)
    return true;
  else if (other.tile_shape_ < tile_shape_)
    return false;
  if (color_shape_ < other.color_shape_)
    return true;
  else if (other.color_shape_ < color_shape_)
    return false;
  if (offsets_ < other.offsets_)
    return true;
  else
    return false;
}

bool Tiling::is_complete_for(const detail::Storage* storage) const
{
  const auto& storage_exts = storage->extents();
  const auto& storage_offs = storage->offsets();

#ifdef DEBUG_LEGATE
  assert(storage_exts.size() == storage_offs.size());
  assert(storage_offs.size() == offsets_.size());
#endif

  uint32_t ndim = storage_exts.size();

  for (uint32_t dim = 0; dim < ndim; ++dim) {
    int64_t my_lo = offsets_[dim];
    int64_t my_hi = my_lo + static_cast<int64_t>(tile_shape_[dim] * color_shape_[dim]);
    if (static_cast<int64_t>(storage_offs[dim]) < my_lo &&
        my_hi < static_cast<int64_t>(storage_offs[dim] + storage_exts[dim]))
      return false;
  }
  return true;
}

bool Tiling::is_disjoint_for(const Domain* launch_domain) const
{
  // TODO: The check really should be that every two points from the launch domain are mapped
  // to two different colors
  return nullptr == launch_domain || launch_domain->get_volume() <= color_shape_.volume();
}

bool Tiling::satisfies_restrictions(const Restrictions& restrictions) const
{
  static auto satisfies_restriction = [](Restriction r, size_t ext) {
    return r != Restriction::FORBID || ext == 1;
  };
  return apply(satisfies_restriction, restrictions, color_shape_).all();
}

Legion::LogicalPartition Tiling::construct(Legion::LogicalRegion region, bool complete) const
{
  auto index_space     = region.get_index_space();
  auto runtime         = detail::Runtime::get_runtime();
  auto part_mgr        = runtime->partition_manager();
  auto index_partition = part_mgr->find_index_partition(index_space, *this);
  if (index_partition != Legion::IndexPartition::NO_PART)
    return runtime->create_logical_partition(region, index_partition);

  auto ndim = static_cast<int32_t>(tile_shape_.size());

  Legion::DomainTransform transform;
  transform.m = ndim;
  transform.n = ndim;
  for (int32_t idx = 0; idx < ndim * ndim; ++idx) transform.matrix[idx] = 0;
  for (int32_t idx = 0; idx < ndim; ++idx) transform.matrix[ndim * idx + idx] = tile_shape_[idx];

  auto extent = to_domain(tile_shape_);
  for (int32_t idx = 0; idx < ndim; ++idx) {
    extent.rect_data[idx] += offsets_[idx];
    extent.rect_data[idx + ndim] += offsets_[idx];
  }

  auto color_domain = to_domain(color_shape_);
  auto color_space  = runtime->find_or_create_index_space(color_domain);

  auto kind = complete ? LEGION_DISJOINT_COMPLETE_KIND : LEGION_DISJOINT_KIND;

  index_partition =
    runtime->create_restricted_partition(index_space, color_space, kind, transform, extent);
  part_mgr->record_index_partition(index_space, *this, index_partition);
  return runtime->create_logical_partition(region, index_partition);
}

bool Tiling::has_launch_domain() const { return true; }

Legion::Domain Tiling::launch_domain() const { return to_domain(color_shape_); }

std::unique_ptr<Partition> Tiling::clone() const { return std::make_unique<Tiling>(*this); }

std::string Tiling::to_string() const
{
  std::stringstream ss;
  ss << "Tiling(tile:" << tile_shape_ << ",colors:" << color_shape_ << ",offset:" << offsets_
     << ")";
  return std::move(ss).str();
}

Shape Tiling::get_child_extents(const Shape& extents, const Shape& color)
{
  auto lo = apply(std::plus<int64_t>{}, tile_shape_ * color, offsets_);
  auto hi = apply(std::plus<int64_t>{}, tile_shape_ * (color + 1), offsets_);
  lo      = apply([](int64_t v) { return std::max<int64_t>(0, v); }, lo);
  hi      = apply([](size_t a, int64_t b) { return std::min<int64_t>(a, b); }, extents, hi);
  return apply([](int64_t h, int64_t l) { return static_cast<size_t>(h - l); }, hi, lo);
}

Shape Tiling::get_child_offsets(const Shape& color)
{
  return apply([](size_t a, int64_t b) { return static_cast<size_t>(static_cast<int64_t>(a) + b); },
               tile_shape_ * color,
               offsets_);
}

Weighted::Weighted(const Legion::FutureMap& weights, const Domain& color_domain)
  : weights_(weights), color_domain_(color_domain), color_shape_(from_domain(color_domain))
{
}

bool Weighted::operator==(const Weighted& other) const
{
  // Since both color_domain_ and color_shape_ are derived from weights_, they don't need to
  // be compared
  return weights_ == other.weights_;
}

bool Weighted::operator<(const Weighted& other) const { return weights_ < other.weights_; }

bool Weighted::is_complete_for(const detail::Storage*) const
{
  // Partition-by-weight partitions are complete by definition
  return true;
}

bool Weighted::is_disjoint_for(const Domain* launch_domain) const
{
  // TODO: The check really should be that every two points from the launch domain are mapped
  // to two different colors
  return nullptr == launch_domain || launch_domain->get_volume() <= color_domain_.get_volume();
}

bool Weighted::satisfies_restrictions(const Restrictions& restrictions) const
{
  static auto satisfies_restriction = [](Restriction r, size_t ext) {
    return r != Restriction::FORBID || ext == 1;
  };
  return apply(satisfies_restriction, restrictions, color_shape_).all();
}

Legion::LogicalPartition Weighted::construct(Legion::LogicalRegion region, bool) const
{
  auto runtime  = detail::Runtime::get_runtime();
  auto part_mgr = runtime->partition_manager();

  const auto& index_space = region.get_index_space();
  auto index_partition    = part_mgr->find_index_partition(index_space, *this);
  if (index_partition != Legion::IndexPartition::NO_PART)
    return runtime->create_logical_partition(region, index_partition);

  auto color_space = runtime->find_or_create_index_space(color_domain_);
  index_partition  = runtime->create_weighted_partition(index_space, color_space, weights_);
  part_mgr->record_index_partition(index_space, *this, index_partition);
  return runtime->create_logical_partition(region, index_partition);
}

bool Weighted::has_launch_domain() const { return true; }

Domain Weighted::launch_domain() const { return color_domain_; }

std::unique_ptr<Partition> Weighted::clone() const
{
  return create_weighted(weights_, color_domain_);
}

std::string Weighted::to_string() const
{
  std::stringstream ss;
  ss << "Weighted({";
  for (Domain::DomainPointIterator it(color_domain_); it; ++it) {
    auto& p = *it;
    ss << p << ":" << weights_.get_result<size_t>(p) << ",";
  }
  ss << "})";
  return std::move(ss).str();
}

Image::Image(std::shared_ptr<detail::LogicalStore> func, std::shared_ptr<Partition> func_partition)
  : func_(std::move(func)), func_partition_(std::move(func_partition))
{
}

bool Image::operator==(const Image& other) const
{
  // FIXME: This needs to be implemented to cache image partitions
  return false;
}

bool Image::operator<(const Image& other) const
{
  // FIXME: This needs to be implemented to cache image partitions
  return false;
}

bool Image::is_complete_for(const detail::Storage* storage) const
{
  // Completeness check for image partitions is expensive, so we give a sound answer
  return false;
}

bool Image::is_disjoint_for(const Domain* launch_domain) const
{
  // Disjointness check for image partitions is expensive, so we give a sound answer;
  return false;
}

bool Image::satisfies_restrictions(const Restrictions& restrictions) const
{
  static auto satisfies_restriction = [](Restriction r, size_t ext) {
    return r != Restriction::FORBID || ext == 1;
  };
  return apply(satisfies_restriction, restrictions, color_shape()).all();
}

Legion::LogicalPartition Image::construct(Legion::LogicalRegion region, bool complete) const
{
  if (!has_launch_domain()) { return Legion::LogicalPartition::NO_PART; }
  auto* func_rf    = func_->get_region_field();
  auto func_region = func_rf->region();
  auto func_partition =
    func_partition_->construct(func_region, func_partition_->is_complete_for(func_->get_storage()));

  auto runtime  = detail::Runtime::get_runtime();
  auto part_mgr = runtime->partition_manager();

  auto target = region.get_index_space();
  auto index_partition =
    part_mgr->find_image_partition(target, func_partition, func_rf->field_id());

  if (Legion::IndexPartition::NO_PART == index_partition) {
    bool is_range    = func_->type()->code == Type::Code::STRUCT;
    auto color_space = runtime->find_or_create_index_space(to_domain(color_shape()));

    auto field_id   = func_rf->field_id();
    index_partition = runtime->create_image_partition(
      target, color_space, func_region, func_partition, field_id, is_range);
    part_mgr->record_image_partition(target, func_partition, field_id, index_partition);
    func_rf->add_invalidation_callback([target, func_partition, field_id]() {
      auto part_mgr = detail::Runtime::get_runtime()->partition_manager();
      part_mgr->invalidate_image_partition(target, func_partition, field_id);
    });
  }

  return runtime->create_logical_partition(region, index_partition);
}

bool Image::has_launch_domain() const { return func_partition_->has_launch_domain(); }

Domain Image::launch_domain() const { return func_partition_->launch_domain(); }

std::unique_ptr<Partition> Image::clone() const { return std::make_unique<Image>(*this); }

std::string Image::to_string() const
{
  std::stringstream ss;
  ss << "Image(func: " << func_->to_string() << ", partition: " << func_partition_->to_string()
     << ")";
  return std::move(ss).str();
}

const Shape& Image::color_shape() const { return func_partition_->color_shape(); }

std::unique_ptr<NoPartition> create_no_partition() { return std::make_unique<NoPartition>(); }

std::unique_ptr<Tiling> create_tiling(Shape&& tile_shape,
                                      Shape&& color_shape,
                                      tuple<int64_t>&& offsets /*= {}*/)
{
  return std::make_unique<Tiling>(
    std::move(tile_shape), std::move(color_shape), std::move(offsets));
}

std::unique_ptr<Weighted> create_weighted(const Legion::FutureMap& weights,
                                          const Domain& color_domain)
{
  return std::make_unique<Weighted>(weights, color_domain);
}

std::unique_ptr<Image> create_image(std::shared_ptr<detail::LogicalStore> func,
                                    std::shared_ptr<Partition> func_partition)
{
  return std::make_unique<Image>(std::move(func), std::move(func_partition));
}

std::ostream& operator<<(std::ostream& out, const Partition& partition)
{
  out << partition.to_string();
  return out;
}

}  // namespace legate
