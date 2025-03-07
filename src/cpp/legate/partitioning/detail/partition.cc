/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partition.h>

#include <legate/data/detail/logical_store.h>
#include <legate/runtime/detail/partition_manager.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <functional>
#include <stdexcept>

namespace legate::detail {

bool NoPartition::is_disjoint_for(const Domain& launch_domain) const
{
  return !launch_domain.is_valid() || launch_domain.get_volume() == 1;
}

InternalSharedPtr<Partition> NoPartition::scale(const tuple<std::uint64_t>& /*factors*/) const
{
  return create_no_partition();
}

InternalSharedPtr<Partition> NoPartition::bloat(const tuple<std::uint64_t>& /*low_offsts*/,
                                                const tuple<std::uint64_t>& /*high_offsets*/) const
{
  return create_no_partition();
}

Legion::Domain NoPartition::launch_domain() const
{
  throw TracedException<std::invalid_argument>{"NoPartition has no launch domain"};
}

std::string NoPartition::to_string() const { return "NoPartition"; }

InternalSharedPtr<Partition> NoPartition::convert(const InternalSharedPtr<Partition>& self,
                                                  const TransformStack* /*transform*/) const
{
  return self;
}

InternalSharedPtr<Partition> NoPartition::invert(const InternalSharedPtr<Partition>& self,
                                                 const TransformStack* /*transform*/) const
{
  return self;
}

// ==========================================================================================

Tiling::Tiling(tuple<std::uint64_t> tile_shape,
               tuple<std::uint64_t> color_shape,
               tuple<std::int64_t> offsets)
  : tile_shape_{std::move(tile_shape)},
    color_shape_{std::move(color_shape)},
    offsets_{offsets.empty() ? legate::full<std::int64_t>(tile_shape_.size(), 0)
                             : std::move(offsets)},
    strides_{tile_shape_}
{
  LEGATE_CHECK(tile_shape_.size() == color_shape_.size());
  LEGATE_CHECK(tile_shape_.size() == offsets_.size());
}

Tiling::Tiling(tuple<std::uint64_t> tile_shape,
               tuple<std::uint64_t> color_shape,
               tuple<std::int64_t> offsets,
               tuple<std::uint64_t> strides)
  : overlapped_{!strides.greater_equal(tile_shape)},
    tile_shape_{std::move(tile_shape)},
    color_shape_{std::move(color_shape)},
    offsets_{offsets.empty() ? legate::full<std::int64_t>(tile_shape_.size(), 0)
                             : std::move(offsets)},
    strides_{std::move(strides)}
{
  LEGATE_CHECK(tile_shape_.size() == color_shape_.size());
  LEGATE_CHECK(tile_shape_.size() == offsets_.size());
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

  LEGATE_ASSERT(storage_exts.size() == storage_offs.size());
  LEGATE_ASSERT(storage_offs.size() == offsets_.size());

  const auto zip =
    legate::detail::zip_equal(offsets_, strides_, color_shape_, storage_offs, storage_exts);
  using zipper_type = std::tuple<const std::int64_t&,
                                 const std::uint64_t&,
                                 const std::uint64_t&,
                                 const std::int64_t&,
                                 const std::uint64_t&>;
  static_assert(std::is_same_v<zipper_type, decltype(*zip.begin())>);

  return std::all_of(zip.begin(), zip.end(), [](const zipper_type& zip_tuple) {
    auto&& [offset, stride, color_shape, storage_off, storage_ext] = zip_tuple;
    const auto my_lo                                               = offset;
    const auto my_hi = my_lo + static_cast<std::int64_t>(stride * color_shape);
    const auto soff  = static_cast<std::int64_t>(storage_off);

    return soff >= my_lo || my_hi >= (soff + static_cast<std::int64_t>(storage_ext));
  });
}

bool Tiling::is_disjoint_for(const Domain& launch_domain) const
{
  // TODO(wonchanl): The check really should be that every two points from the launch domain are
  // mapped to two different colors
  return !overlapped_ &&
         (!launch_domain.is_valid() || launch_domain.get_volume() <= color_shape_.volume());
}

bool Tiling::satisfies_restrictions(const Restrictions& restrictions) const
{
  constexpr auto satisfies_restriction = [](Restriction r, std::size_t ext) {
    return r != Restriction::FORBID || ext == 1;
  };
  return apply(satisfies_restriction, restrictions, color_shape_).all();
}

InternalSharedPtr<Partition> Tiling::scale(const tuple<std::uint64_t>& factors) const
{
  auto new_offsets = apply(
    [](std::int64_t off, std::size_t factor) { return off * static_cast<std::int64_t>(factor); },
    offsets_,
    factors);
  return create_tiling(
    tile_shape_ * factors, tuple<std::uint64_t>{color_shape_}, std::move(new_offsets));
}

InternalSharedPtr<Partition> Tiling::bloat(const tuple<std::uint64_t>& low_offsets,
                                           const tuple<std::uint64_t>& high_offsets) const
{
  auto tile_shape = tile_shape_ + low_offsets + high_offsets;
  auto offsets =
    apply([](std::int64_t off, std::size_t diff) { return off - static_cast<std::int64_t>(diff); },
          offsets_,
          low_offsets);

  return create_tiling(std::move(tile_shape),
                       tuple<std::uint64_t>{color_shape_},
                       std::move(offsets),
                       tuple<std::uint64_t>{tile_shape_});
}

Legion::LogicalPartition Tiling::construct(Legion::LogicalRegion region, bool complete) const
{
  auto&& index_space   = region.get_index_space();
  auto runtime         = detail::Runtime::get_runtime();
  auto* part_mgr       = runtime->partition_manager();
  auto index_partition = part_mgr->find_index_partition(index_space, *this);

  if (index_partition != Legion::IndexPartition::NO_PART) {
    return runtime->create_logical_partition(region, index_partition);
  }

  const auto ndim = static_cast<std::int32_t>(tile_shape_.size());
  Legion::DomainTransform transform;

  transform.m = ndim;
  transform.n = ndim;
  for (std::int32_t idx = 0; idx < ndim * ndim; ++idx) {
    transform.matrix[idx] = 0;
  }
  for (std::int32_t idx = 0; idx < ndim; ++idx) {
    transform.matrix[(ndim * idx) + idx] = static_cast<Legion::coord_t>(strides_[idx]);
  }

  auto extent = detail::to_domain(tile_shape_);
  for (std::int32_t idx = 0; idx < ndim; ++idx) {
    extent.rect_data[idx] += offsets_[idx];
    extent.rect_data[idx + ndim] += offsets_[idx];
  }

  auto&& color_space = runtime->find_or_create_index_space(color_shape_);
  const auto kind    = complete ? LEGION_DISJOINT_COMPLETE_KIND : LEGION_DISJOINT_KIND;

  index_partition =
    runtime->create_restricted_partition(index_space, color_space, kind, transform, extent);
  part_mgr->record_index_partition(index_space, *this, index_partition);
  return runtime->create_logical_partition(region, index_partition);
}

Legion::Domain Tiling::launch_domain() const { return detail::to_domain(color_shape_); }

std::string Tiling::to_string() const
{
  return fmt::format("Tiling(tile:{},colors:{},offset:{},strides:{})",
                     tile_shape_,
                     color_shape_,
                     offsets_,
                     strides_);
}

InternalSharedPtr<Partition> Tiling::convert(const InternalSharedPtr<Partition>& self,
                                             const TransformStack* transform) const
{
  if (transform->identity()) {
    return self;
  }
  return create_tiling(transform->convert_extents(tile_shape()),
                       transform->convert_color_shape(color_shape()),
                       transform->convert_point(offsets()),
                       transform->convert_extents(strides_));
}

InternalSharedPtr<Partition> Tiling::invert(const InternalSharedPtr<Partition>& self,
                                            const TransformStack* transform) const
{
  if (transform->identity()) {
    return self;
  }
  return create_tiling(transform->invert_extents(tile_shape()),
                       transform->invert_color_shape(color_shape()),
                       transform->invert_point(offsets()),
                       transform->invert_extents(strides_));
}

tuple<std::uint64_t> Tiling::get_child_extents(const tuple<std::uint64_t>& extents,
                                               const tuple<std::uint64_t>& color) const
{
  auto lo = apply(std::plus<std::int64_t>{}, tile_shape_ * color, offsets_);
  auto hi = apply(std::plus<std::int64_t>{}, tile_shape_ * (color + 1), offsets_);
  lo      = apply([](std::int64_t v) { return std::max(static_cast<std::int64_t>(0), v); }, lo);
  hi =
    apply([](std::size_t a, std::int64_t b) { return std::min(static_cast<std::int64_t>(a), b); },
          extents,
          hi);
  return apply(
    [](std::int64_t h, std::int64_t l) { return static_cast<std::uint64_t>(h - l); }, hi, lo);
}

tuple<std::int64_t> Tiling::get_child_offsets(const tuple<std::uint64_t>& color) const
{
  return apply([](std::uint64_t a, std::int64_t b) { return static_cast<std::int64_t>(a) + b; },
               strides_ * color,
               offsets_);
}

std::size_t Tiling::hash() const { return hash_all(tile_shape_, color_shape_, offsets_, strides_); }

// ==========================================================================================

Weighted::Weighted(Legion::FutureMap weights, const Domain& color_domain)
  : weights_{std::move(weights)},
    color_domain_{color_domain},
    color_shape_{detail::from_domain(color_domain)}
{
}

Weighted::~Weighted()
{
  if (detail::has_started() || !weights_.exists()) {
    return;
  }
  // FIXME: Leak the FutureMap handle if the runtime has already shut down, as there's no hope
  // that this would be collected by the Legion runtime
  static_cast<void>(std::make_unique<Legion::FutureMap>(std::move(weights_)).release());
}  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)

bool Weighted::operator==(const Weighted& other) const
{
  // Since both color_domain_ and color_shape_ are derived from weights_, they don't need to
  // be compared
  return weights_ == other.weights_;
}

bool Weighted::operator<(const Weighted& other) const { return weights_ < other.weights_; }

bool Weighted::is_disjoint_for(const Domain& launch_domain) const
{
  // TODO(wonchanl): The check really should be that every two points from the launch domain are
  // mapped to two different colors
  return !launch_domain.is_valid() || launch_domain.get_volume() <= color_domain_.get_volume();
}

bool Weighted::satisfies_restrictions(const Restrictions& restrictions) const
{
  constexpr auto satisfies_restriction = [](Restriction r, std::size_t ext) {
    return r != Restriction::FORBID || ext == 1;
  };
  return apply(satisfies_restriction, restrictions, color_shape_).all();
}

InternalSharedPtr<Partition> Weighted::scale(const tuple<std::uint64_t>& /*factors*/) const
{
  throw TracedException<std::runtime_error>{"Not implemented"};
  return {};
}

InternalSharedPtr<Partition> Weighted::bloat(const tuple<std::uint64_t>& /*low_offsts*/,
                                             const tuple<std::uint64_t>& /*high_offsets*/) const
{
  throw TracedException<std::runtime_error>{"Not implemented"};
  return {};
}

Legion::LogicalPartition Weighted::construct(Legion::LogicalRegion region, bool) const
{
  auto runtime            = detail::Runtime::get_runtime();
  auto* part_mgr          = runtime->partition_manager();
  const auto& index_space = region.get_index_space();
  auto index_partition    = part_mgr->find_index_partition(index_space, *this);

  if (index_partition != Legion::IndexPartition::NO_PART) {
    return runtime->create_logical_partition(region, index_partition);
  }

  auto&& color_space = runtime->find_or_create_index_space(color_shape_);

  index_partition = runtime->create_weighted_partition(index_space, color_space, weights_);
  part_mgr->record_index_partition(index_space, *this, index_partition);
  return runtime->create_logical_partition(region, index_partition);
}

std::string Weighted::to_string() const
{
  std::string result = "Weighted({";

  for (Domain::DomainPointIterator it{color_domain_}; it; ++it) {
    auto& p = *it;

    fmt::format_to(
      std::back_inserter(result), "{}:{},", fmt::streamed(p), weights_.get_result<std::size_t>(p));
  }
  result += "})";
  return result;
}

InternalSharedPtr<Partition> Weighted::convert(const InternalSharedPtr<Partition>& self,
                                               const TransformStack* transform) const
{
  if (transform->identity()) {
    return self;
  }
  throw TracedException<NonInvertibleTransformation>{};
  return nullptr;
}

InternalSharedPtr<Partition> Weighted::invert(const InternalSharedPtr<Partition>& self,
                                              const TransformStack* transform) const
{
  if (transform->identity()) {
    return self;
  }
  auto color_domain = to_domain(transform->invert_color_shape(color_shape_));
  // Weighted partitions are created only for 1D stores. So, if we're here, the 1D store to which
  // this partition is applied would be a degenerate N-D store such that all but one dimension are
  // of extent 1. So, we only need to delinearize the future map holding the weights so the domain
  // matches the color domain.
  return create_weighted(Runtime::get_runtime()->delinearize_future_map(weights_, color_domain),
                         color_domain);
}

// ==========================================================================================

Image::Image(InternalSharedPtr<detail::LogicalStore> func,
             InternalSharedPtr<Partition> func_partition,
             mapping::detail::Machine machine,
             ImageComputationHint hint)
  : func_{std::move(func)},
    func_partition_{std::move(func_partition)},
    machine_{std::move(machine)},
    hint_{hint}
{
}

bool Image::operator==(const Image& other) const
{
  return func_->id() == other.func_->id() && func_partition_ == other.func_partition_ &&
         hint_ == other.hint_;
}

bool Image::is_complete_for(const detail::Storage* /*storage*/) const
{
  // Completeness check for image partitions is expensive, so we give a sound answer
  return false;
}

bool Image::is_disjoint_for(const Domain& launch_domain) const
{
  // Disjointedness check for image partitions is expensive, so we give a sound answer;
  return !launch_domain.is_valid();
}

bool Image::satisfies_restrictions(const Restrictions& restrictions) const
{
  constexpr auto satisfies_restriction = [](Restriction r, std::size_t ext) {
    return r != Restriction::FORBID || ext == 1;
  };
  return apply(satisfies_restriction, restrictions, color_shape()).all();
}

InternalSharedPtr<Partition> Image::scale(const tuple<std::uint64_t>& /*factors*/) const
{
  throw TracedException<std::runtime_error>{"Not implemented"};
  return {};
}

InternalSharedPtr<Partition> Image::bloat(const tuple<std::uint64_t>& /*low_offsts*/,
                                          const tuple<std::uint64_t>& /*high_offsets*/) const
{
  throw TracedException<std::runtime_error>{"Not implemented"};
  return {};
}

Legion::LogicalPartition Image::construct(Legion::LogicalRegion region, bool /*complete*/) const
{
  if (!has_launch_domain()) {
    return Legion::LogicalPartition::NO_PART;
  }

  auto&& func_rf      = func_->get_region_field();
  auto&& func_region  = func_rf->region();
  auto func_partition = func_partition_->construct(
    func_region, func_partition_->is_complete_for(func_->get_storage().get()));

  auto runtime   = detail::Runtime::get_runtime();
  auto* part_mgr = runtime->partition_manager();

  auto target          = region.get_index_space();
  const auto field_id  = func_rf->field_id();
  auto index_partition = part_mgr->find_image_partition(target, func_partition, field_id, hint_);

  if (Legion::IndexPartition::NO_PART == index_partition) {
    auto construct_image_partition = [&] {
      switch (hint_) {
        case ImageComputationHint::NO_HINT: {
          const bool is_range = func_->type()->code == Type::Code::STRUCT;
          auto color_space    = runtime->find_or_create_index_space(color_shape());
          return runtime->create_image_partition(
            target, color_space, func_region, func_partition, field_id, is_range, machine_);
        }
        case ImageComputationHint::MIN_MAX: {
          return runtime->create_approximate_image_partition(func_, func_partition_, target, false);
        }
        case ImageComputationHint::FIRST_LAST: {
          return runtime->create_approximate_image_partition(func_, func_partition_, target, true);
        }
      }
      LEGATE_UNREACHABLE();
    };

    index_partition = construct_image_partition();
    part_mgr->record_image_partition(target, func_partition, field_id, hint_, index_partition);
    func_rf->add_invalidation_callback([target, func_partition, field_id, hint = hint_]() noexcept {
      detail::Runtime::get_runtime()->partition_manager()->invalidate_image_partition(
        target, func_partition, field_id, hint);
    });
  }

  return runtime->create_logical_partition(region, index_partition);
}

bool Image::has_launch_domain() const { return func_partition_->has_launch_domain(); }

Domain Image::launch_domain() const { return func_partition_->launch_domain(); }

std::string Image::to_string() const
{
  return fmt::format("Image(func: {}, partition: {}, hint: {})",
                     func_->to_string(),
                     func_partition_->to_string(),
                     hint_);
}

const tuple<std::uint64_t>& Image::color_shape() const { return func_partition_->color_shape(); }

InternalSharedPtr<Partition> Image::convert(const InternalSharedPtr<Partition>& self,
                                            const detail::TransformStack* transform) const
{
  if (transform->identity()) {
    return self;
  }
  throw TracedException<NonInvertibleTransformation>{};
  return nullptr;
}

InternalSharedPtr<Partition> Image::invert(const InternalSharedPtr<Partition>& self,
                                           const detail::TransformStack* transform) const
{
  if (transform->identity()) {
    return self;
  }
  throw TracedException<NonInvertibleTransformation>{};
  return nullptr;
}

InternalSharedPtr<NoPartition> create_no_partition()
{
  static auto result = make_internal_shared<NoPartition>();
  return result;
}

InternalSharedPtr<Tiling> create_tiling(tuple<std::uint64_t> tile_shape,
                                        tuple<std::uint64_t> color_shape,
                                        tuple<std::int64_t> offsets /*= {}*/)
{
  return make_internal_shared<Tiling>(
    std::move(tile_shape), std::move(color_shape), std::move(offsets));
}

InternalSharedPtr<Tiling> create_tiling(tuple<std::uint64_t> tile_shape,
                                        tuple<std::uint64_t> color_shape,
                                        tuple<std::int64_t> offsets,
                                        tuple<std::uint64_t> strides)
{
  return make_internal_shared<Tiling>(
    std::move(tile_shape), std::move(color_shape), std::move(offsets), std::move(strides));
}

InternalSharedPtr<Weighted> create_weighted(const Legion::FutureMap& weights,
                                            const Domain& color_domain)
{
  return make_internal_shared<Weighted>(weights, color_domain);
}

InternalSharedPtr<Image> create_image(InternalSharedPtr<detail::LogicalStore> func,
                                      InternalSharedPtr<Partition> func_partition,
                                      mapping::detail::Machine machine,
                                      ImageComputationHint hint)
{
  return make_internal_shared<Image>(
    std::move(func), std::move(func_partition), std::move(machine), hint);
}

std::ostream& operator<<(std::ostream& out, const Partition& partition)
{
  out << partition.to_string();
  return out;
}

}  // namespace legate::detail
