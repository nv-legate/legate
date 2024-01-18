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

#include "core/partitioning/partition.h"

#include "core/data/detail/logical_store.h"
#include "core/runtime/detail/partition_manager.h"
#include "core/runtime/detail/runtime.h"
#include "core/type/detail/type_info.h"
#include "core/utilities/detail/tuple.h"

#include <functional>
#include <sstream>

namespace legate {

bool NoPartition::is_disjoint_for(const Domain& launch_domain) const
{
  return !launch_domain.is_valid() || launch_domain.get_volume() == 1;
}

std::unique_ptr<Partition> NoPartition::scale(const tuple<uint64_t>& /*factors*/) const
{
  return create_no_partition();
}

std::unique_ptr<Partition> NoPartition::bloat(const tuple<uint64_t>& /*low_offsts*/,
                                              const tuple<uint64_t>& /*high_offsets*/) const
{
  return create_no_partition();
}

Legion::Domain NoPartition::launch_domain() const
{
  assert(false);
  return {};
}

std::unique_ptr<Partition> NoPartition::clone() const { return create_no_partition(); }

std::string NoPartition::to_string() const { return "NoPartition"; }

Tiling::Tiling(tuple<uint64_t>&& tile_shape,
               tuple<uint64_t>&& color_shape,
               tuple<int64_t>&& offsets)
  : tile_shape_{std::move(tile_shape)},
    color_shape_{std::move(color_shape)},
    offsets_{offsets.empty() ? legate::full<int64_t>(tile_shape_.size(), 0) : std::move(offsets)},
    strides_{tile_shape_}
{
  assert(tile_shape_.size() == color_shape_.size());
  assert(tile_shape_.size() == offsets_.size());
}

Tiling::Tiling(tuple<uint64_t>&& tile_shape,
               tuple<uint64_t>&& color_shape,
               tuple<int64_t>&& offsets,
               tuple<uint64_t>&& strides)
  : overlapped_{strides < tile_shape},
    tile_shape_{std::move(tile_shape)},
    color_shape_{std::move(color_shape)},
    offsets_{offsets.empty() ? legate::full<int64_t>(tile_shape_.size(), 0) : std::move(offsets)},
    strides_{std::move(strides)}
{
  if (!overlapped_) {
    throw std::invalid_argument("This constructor must be called only for overlapped tiling");
  }
  assert(tile_shape_.size() == color_shape_.size());
  assert(tile_shape_.size() == offsets_.size());
}

bool Tiling::operator==(const Tiling& other) const
{
  return tile_shape_ == other.tile_shape_ && color_shape_ == other.color_shape_ &&
         offsets_ == other.offsets_ && strides_ == other.strides_;
}

bool Tiling::is_complete_for(const detail::Storage* storage) const
{
  const auto& storage_exts = storage->extents();
  const auto& storage_offs = storage->offsets();

  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(storage_exts.size() == storage_offs.size());
    assert(storage_offs.size() == offsets_.size());
  }

  const auto ndim = static_cast<uint32_t>(storage_exts.size());

  for (uint32_t dim = 0; dim < ndim; ++dim) {
    const int64_t my_lo = offsets_[dim];
    const int64_t my_hi = my_lo + static_cast<int64_t>(strides_[dim] * color_shape_[dim]);
    const auto soff     = static_cast<int64_t>(storage_offs[dim]);

    if (soff < my_lo && my_hi < (soff + static_cast<int64_t>(storage_exts[dim]))) {
      return false;
    }
  }
  return true;
}

bool Tiling::is_disjoint_for(const Domain& launch_domain) const
{
  // TODO: The check really should be that every two points from the launch domain are mapped
  // to two different colors
  return !overlapped_ &&
         (!launch_domain.is_valid() || launch_domain.get_volume() <= color_shape_.volume());
}

bool Tiling::satisfies_restrictions(const Restrictions& restrictions) const
{
  static auto satisfies_restriction = [](Restriction r, size_t ext) {
    return r != Restriction::FORBID || ext == 1;
  };
  return apply(satisfies_restriction, restrictions, color_shape_).all();
}

std::unique_ptr<Partition> Tiling::scale(const tuple<uint64_t>& factors) const
{
  auto new_offsets =
    apply([](int64_t off, size_t factor) { return off * static_cast<int64_t>(factor); },
          offsets_,
          factors);
  return create_tiling(
    tile_shape_ * factors, tuple<uint64_t>{color_shape_}, std::move(new_offsets));
}

std::unique_ptr<Partition> Tiling::bloat(const tuple<uint64_t>& low_offsets,
                                         const tuple<uint64_t>& high_offsets) const
{
  auto tile_shape = tile_shape_ + low_offsets + high_offsets;
  auto offsets    = apply([](int64_t off, size_t diff) { return off - static_cast<int64_t>(diff); },
                       offsets_,
                       low_offsets);

  return create_tiling(std::move(tile_shape),
                       tuple<uint64_t>{color_shape_},
                       std::move(offsets),
                       tuple<uint64_t>{tile_shape_});
}

Legion::LogicalPartition Tiling::construct(Legion::LogicalRegion region, bool complete) const
{
  auto index_space     = region.get_index_space();
  auto runtime         = detail::Runtime::get_runtime();
  auto part_mgr        = runtime->partition_manager();
  auto index_partition = part_mgr->find_index_partition(index_space, *this);

  if (index_partition != Legion::IndexPartition::NO_PART) {
    return runtime->create_logical_partition(region, index_partition);
  }

  auto ndim = static_cast<int32_t>(tile_shape_.size());

  Legion::DomainTransform transform;
  transform.m = ndim;
  transform.n = ndim;
  for (int32_t idx = 0; idx < ndim * ndim; ++idx) {
    transform.matrix[idx] = 0;
  }
  for (int32_t idx = 0; idx < ndim; ++idx) {
    transform.matrix[ndim * idx + idx] = static_cast<Legion::coord_t>(strides_[idx]);
  }

  auto extent = detail::to_domain(tile_shape_);
  for (int32_t idx = 0; idx < ndim; ++idx) {
    extent.rect_data[idx] += offsets_[idx];
    extent.rect_data[idx + ndim] += offsets_[idx];
  }

  auto color_space = runtime->find_or_create_index_space(color_shape_);

  auto kind = complete ? LEGION_DISJOINT_COMPLETE_KIND : LEGION_DISJOINT_KIND;

  index_partition =
    runtime->create_restricted_partition(index_space, color_space, kind, transform, extent);
  part_mgr->record_index_partition(index_space, *this, index_partition);
  return runtime->create_logical_partition(region, index_partition);
}

Legion::Domain Tiling::launch_domain() const { return detail::to_domain(color_shape_); }

std::unique_ptr<Partition> Tiling::clone() const { return std::make_unique<Tiling>(*this); }

std::string Tiling::to_string() const
{
  std::stringstream ss;

  ss << "Tiling(tile:" << tile_shape_ << ",colors:" << color_shape_ << ",offset:" << offsets_
     << ",strides:" << strides_ << ")";
  return std::move(ss).str();
}

tuple<uint64_t> Tiling::get_child_extents(const tuple<uint64_t>& extents,
                                          const tuple<uint64_t>& color) const
{
  auto lo = apply(std::plus<int64_t>{}, tile_shape_ * color, offsets_);
  auto hi = apply(std::plus<int64_t>{}, tile_shape_ * (color + 1), offsets_);
  lo      = apply([](int64_t v) { return std::max(static_cast<int64_t>(0), v); }, lo);
  hi = apply([](size_t a, int64_t b) { return std::min(static_cast<int64_t>(a), b); }, extents, hi);
  return apply([](int64_t h, int64_t l) { return static_cast<uint64_t>(h - l); }, hi, lo);
}

tuple<uint64_t> Tiling::get_child_offsets(const tuple<uint64_t>& color) const
{
  return apply(
    [](uint64_t a, int64_t b) { return static_cast<uint64_t>(static_cast<int64_t>(a) + b); },
    strides_ * color,
    offsets_);
}

size_t Tiling::hash() const { return hash_all(tile_shape_, color_shape_, offsets_, strides_); }

Weighted::Weighted(const Legion::FutureMap& weights, const Domain& color_domain)
  : weights_{std::make_unique<Legion::FutureMap>(weights)},
    color_domain_{color_domain},
    color_shape_{detail::from_domain(color_domain)}
{
}

Weighted::~Weighted()
{
  if (!detail::Runtime::get_runtime()->initialized()) {
    // FIXME: Leak the FutureMap handle if the runtime has already shut down, as there's no hope
    // that this would be collected by the Legion runtime
    static_cast<void>(weights_.release());
  }
}

Weighted::Weighted(const Weighted& other)
  : weights_{std::make_unique<Legion::FutureMap>(*other.weights_)},
    color_domain_{other.color_domain_},
    color_shape_{other.color_shape_}
{
}

bool Weighted::operator==(const Weighted& other) const
{
  // Since both color_domain_ and color_shape_ are derived from weights_, they don't need to
  // be compared
  return *weights_ == *other.weights_;
}

bool Weighted::operator<(const Weighted& other) const { return *weights_ < *other.weights_; }

bool Weighted::is_disjoint_for(const Domain& launch_domain) const
{
  // TODO: The check really should be that every two points from the launch domain are mapped
  // to two different colors
  return !launch_domain.is_valid() || launch_domain.get_volume() <= color_domain_.get_volume();
}

bool Weighted::satisfies_restrictions(const Restrictions& restrictions) const
{
  static auto satisfies_restriction = [](Restriction r, size_t ext) {
    return r != Restriction::FORBID || ext == 1;
  };
  return apply(satisfies_restriction, restrictions, color_shape_).all();
}

std::unique_ptr<Partition> Weighted::scale(const tuple<uint64_t>& /*factors*/) const
{
  throw std::runtime_error{"Not implemented"};
  return {};
}

std::unique_ptr<Partition> Weighted::bloat(const tuple<uint64_t>& /*low_offsts*/,
                                           const tuple<uint64_t>& /*high_offsets*/) const
{
  throw std::runtime_error{"Not implemented"};
  return {};
}

Legion::LogicalPartition Weighted::construct(Legion::LogicalRegion region, bool) const
{
  auto runtime  = detail::Runtime::get_runtime();
  auto part_mgr = runtime->partition_manager();

  const auto& index_space = region.get_index_space();
  auto index_partition    = part_mgr->find_index_partition(index_space, *this);

  if (index_partition != Legion::IndexPartition::NO_PART) {
    return runtime->create_logical_partition(region, index_partition);
  }

  auto color_space = runtime->find_or_create_index_space(color_shape_);

  index_partition = runtime->create_weighted_partition(index_space, color_space, *weights_);
  part_mgr->record_index_partition(index_space, *this, index_partition);
  return runtime->create_logical_partition(region, index_partition);
}

std::unique_ptr<Partition> Weighted::clone() const
{
  return create_weighted(*weights_, color_domain_);
}

std::string Weighted::to_string() const
{
  std::stringstream ss;

  ss << "Weighted({";
  for (Domain::DomainPointIterator it{color_domain_}; it; ++it) {
    auto& p = *it;
    ss << p << ":" << weights_->get_result<size_t>(p) << ",";
  }
  ss << "})";
  return std::move(ss).str();
}

Image::Image(InternalSharedPtr<detail::LogicalStore> func,
             InternalSharedPtr<Partition> func_partition,
             mapping::detail::Machine machine)
  : func_{std::move(func)}, func_partition_{std::move(func_partition)}, machine_{std::move(machine)}
{
}

bool Image::operator==(const Image& /*other*/) const
{
  // FIXME: This needs to be implemented to cache image partitions
  return false;
}

bool Image::operator<(const Image& /*other*/) const
{
  // FIXME: This needs to be implemented to cache image partitions
  return false;
}

bool Image::is_complete_for(const detail::Storage* /*storage*/) const
{
  // Completeness check for image partitions is expensive, so we give a sound answer
  return false;
}

bool Image::is_disjoint_for(const Domain& launch_domain) const
{
  // Disjointness check for image partitions is expensive, so we give a sound answer;
  return !launch_domain.is_valid();
}

bool Image::satisfies_restrictions(const Restrictions& restrictions) const
{
  static auto satisfies_restriction = [](Restriction r, size_t ext) {
    return r != Restriction::FORBID || ext == 1;
  };
  return apply(satisfies_restriction, restrictions, color_shape()).all();
}

std::unique_ptr<Partition> Image::scale(const tuple<uint64_t>& /*factors*/) const
{
  throw std::runtime_error{"Not implemented"};
  return {};
}

std::unique_ptr<Partition> Image::bloat(const tuple<uint64_t>& /*low_offsts*/,
                                        const tuple<uint64_t>& /*high_offsets*/) const
{
  throw std::runtime_error{"Not implemented"};
  return {};
}

Legion::LogicalPartition Image::construct(Legion::LogicalRegion region, bool /*complete*/) const
{
  if (!has_launch_domain()) {
    return Legion::LogicalPartition::NO_PART;
  }
  auto func_rf     = func_->get_region_field();
  auto func_region = func_rf->region();
  auto func_partition =
    func_partition_->construct(func_region, func_partition_->is_complete_for(func_->get_storage()));

  auto runtime  = detail::Runtime::get_runtime();
  auto part_mgr = runtime->partition_manager();

  auto target = region.get_index_space();
  auto index_partition =
    part_mgr->find_image_partition(target, func_partition, func_rf->field_id());

  if (Legion::IndexPartition::NO_PART == index_partition) {
    const bool is_range = func_->type()->code == Type::Code::STRUCT;
    auto color_space    = runtime->find_or_create_index_space(color_shape());

    auto field_id   = func_rf->field_id();
    index_partition = runtime->create_image_partition(
      target, color_space, func_region, func_partition, field_id, is_range, machine_);
    part_mgr->record_image_partition(target, func_partition, field_id, index_partition);
    func_rf->add_invalidation_callback([target, func_partition, field_id]() noexcept {
      detail::Runtime::get_runtime()->partition_manager()->invalidate_image_partition(
        target, func_partition, field_id);
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

const tuple<uint64_t>& Image::color_shape() const { return func_partition_->color_shape(); }

std::unique_ptr<NoPartition> create_no_partition() { return std::make_unique<NoPartition>(); }

std::unique_ptr<Tiling> create_tiling(tuple<uint64_t>&& tile_shape,
                                      tuple<uint64_t>&& color_shape,
                                      tuple<int64_t>&& offsets /*= {}*/)
{
  return std::make_unique<Tiling>(
    std::move(tile_shape), std::move(color_shape), std::move(offsets));
}

std::unique_ptr<Tiling> create_tiling(tuple<uint64_t>&& tile_shape,
                                      tuple<uint64_t>&& color_shape,
                                      tuple<int64_t>&& offsets,
                                      tuple<uint64_t>&& strides)
{
  return std::make_unique<Tiling>(
    std::move(tile_shape), std::move(color_shape), std::move(offsets), std::move(strides));
}

std::unique_ptr<Weighted> create_weighted(const Legion::FutureMap& weights,
                                          const Domain& color_domain)
{
  return std::make_unique<Weighted>(weights, color_domain);
}

std::unique_ptr<Image> create_image(InternalSharedPtr<detail::LogicalStore> func,
                                    InternalSharedPtr<Partition> func_partition,
                                    mapping::detail::Machine machine)
{
  return std::make_unique<Image>(std::move(func), std::move(func_partition), std::move(machine));
}

std::ostream& operator<<(std::ostream& out, const Partition& partition)
{
  out << partition.to_string();
  return out;
}

}  // namespace legate
