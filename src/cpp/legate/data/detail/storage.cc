/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/storage.h>

#include <legate_defines.h>

#include <legate/data/detail/shape.h>
#include <legate/data/detail/storage_partition.h>
#include <legate/mapping/detail/machine.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/tuning/parallel_policy.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/type_traits.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace legate::detail {

Storage::Storage(InternalSharedPtr<Shape> shape,
                 std::uint32_t field_size,
                 bool optimize_scalar,
                 std::string_view provenance)
  : storage_id_{Runtime::get_runtime().get_unique_storage_id()},
    unbound_{shape->unbound()},
    shape_{std::move(shape)},
    provenance_{std::move(provenance)},
    offsets_{tags::size_tag, dim(), 0}
{
  // initialize storage_data_, prioritizing an optimized scalar storage but deferring
  // to region fields if unable to do so
  if (optimize_scalar && unbound()) {
    storage_data_.emplace<std::optional<Legion::FutureMap>>(std::nullopt);
  } else if (optimize_scalar && this->shape()->ready() && (this->shape()->volume() == 1)) {
    // Note we do not blindly check the shape volume as it would flush the scheduling window
    storage_data_.emplace<std::optional<Legion::Future>>(std::nullopt);
  } else if (unbound()) {
    storage_data_.emplace<std::optional<InternalSharedPtr<LogicalRegionField>>>(std::nullopt);
  } else {
    auto&& runtime = Runtime::get_runtime();
    storage_data_.emplace<std::optional<InternalSharedPtr<LogicalRegionField>>>(
      runtime.create_region_field(this->shape(), field_size));
    runtime.attach_alloc_info(get_region_field(), this->provenance());
  }

  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

Storage::Storage(InternalSharedPtr<Shape> shape, Legion::Future future, std::string_view provenance)
  : storage_id_{Runtime::get_runtime().get_unique_storage_id()},
    shape_{std::move(shape)},
    provenance_{std::move(provenance)},
    offsets_{legate::full(dim(), std::int64_t{0})},
    storage_data_{std::optional<Legion::Future>{std::move(future)}}
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

Storage::Storage(SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents,
                 InternalSharedPtr<StoragePartition> parent,
                 SmallVector<std::uint64_t, LEGATE_MAX_DIM> color,
                 SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets)
  : storage_id_{Runtime::get_runtime().get_unique_storage_id()},
    shape_{make_internal_shared<Shape>(std::move(extents))},
    level_{parent->level() + 1},
    parent_{std::move(parent)},
    color_{std::move(color)},
    offsets_{std::move(offsets)},
    storage_data_{
      std::optional<InternalSharedPtr<LogicalRegionField>>{(*parent_)->get_child_data(color_)}}
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

// Leak is intentional
// NOLINTBEGIN(clang-analyzer-cplusplus.NewDeleteLeaks)
Storage::~Storage()
{
  if (has_started()) {
    return;
  }

  try {
    std::visit(
      Overload{
        [&](std::optional<Legion::Future>& future) {
          if (future.has_value() && future->exists()) {
            // FIXME: Leak the Future handle if the runtime has already shut down, as there's no
            // hope that this would be collected by the Legion runtime
            static_cast<void>(std::make_unique<Legion::Future>(*std::move(future)).release());
          }
        },
        [&](std::optional<Legion::FutureMap>& future_map) {
          if (future_map.has_value()) {
            static_cast<void>(
              std::make_unique<Legion::FutureMap>(*std::move(future_map)).release());
          }
        },
        [&](std::optional<InternalSharedPtr<LogicalRegionField>>&) {
          // Do nothing
        },
      },
      storage_data_);
  } catch (const std::exception& e) {
    LEGATE_ABORT(e.what());
  }
}

// NOLINTEND(clang-analyzer-cplusplus.NewDeleteLeaks)

Storage::Kind Storage::kind() const
{
  return std::visit(
    Overload{
      [&](const std::optional<InternalSharedPtr<LogicalRegionField>>&) -> Storage::Kind {
        return Kind::REGION_FIELD;
      },
      [&](const std::optional<Legion::Future>&) -> Storage::Kind { return Kind::FUTURE; },
      [&](const std::optional<Legion::FutureMap>&) -> Storage::Kind { return Kind::FUTURE_MAP; },
    },
    storage_data_);
}

// Private getter implementations
const std::optional<InternalSharedPtr<LogicalRegionField>>& Storage::region_field_() const
{
  return std::get<std::optional<InternalSharedPtr<LogicalRegionField>>>(storage_data_);
}

std::optional<InternalSharedPtr<LogicalRegionField>>& Storage::region_field_()
{
  return std::get<std::optional<InternalSharedPtr<LogicalRegionField>>>(storage_data_);
}

const std::optional<Legion::Future>& Storage::future_() const
{
  return std::get<std::optional<Legion::Future>>(storage_data_);
}

std::optional<Legion::Future>& Storage::future_()
{
  return std::get<std::optional<Legion::Future>>(storage_data_);
}

const std::optional<Legion::FutureMap>& Storage::future_map_() const
{
  return std::get<std::optional<Legion::FutureMap>>(storage_data_);
}

std::optional<Legion::FutureMap>& Storage::future_map_()
{
  return std::get<std::optional<Legion::FutureMap>>(storage_data_);
}

Span<const std::int64_t> Storage::offsets() const
{
  LEGATE_ASSERT(!unbound_);
  return offsets_;
}

bool Storage::overlaps(const InternalSharedPtr<Storage>& other) const
{
  const auto* lhs = this;
  const auto* rhs = other.get();

  if (lhs == rhs) {
    return true;
  }

  if (lhs->get_root() != rhs->get_root()) {
    return false;
  }

  if (lhs->volume() == 0 || rhs->volume() == 0) {
    return false;
  }

  auto&& lexts = lhs->extents();
  auto&& rexts = rhs->extents();

  for (std::uint32_t idx = 0; idx < dim(); ++idx) {
    auto lext = static_cast<std::int64_t>(lexts[idx]);
    auto rext = static_cast<std::int64_t>(rexts[idx]);
    auto loff = lhs->offsets_[idx];
    auto roff = rhs->offsets_[idx];

    if (loff <= roff ? roff < loff + lext : loff < roff + rext) {
      continue;
    }
    return false;
  }
  return true;
}

namespace {

class EqualVisitor {
 public:
  bool operator()(const std::optional<InternalSharedPtr<LogicalRegionField>>& lhs_rf,
                  const std::optional<InternalSharedPtr<LogicalRegionField>>& rhs_rf) const
  {
    if (lhs_rf.has_value() != rhs_rf.has_value()) {
      return false;
    }
    if (!lhs_rf.has_value() && !rhs_rf.has_value()) {
      return false;
    }
    // Both region fields exist, check if they are the same
    return (*lhs_rf)->region() == (*rhs_rf)->region() &&
           (*lhs_rf)->field_id() == (*rhs_rf)->field_id();
  }

  bool operator()(const std::optional<Legion::Future>&, const std::optional<Legion::Future>&) const
  {
    return false;
  }

  bool operator()(const std::optional<Legion::FutureMap>&,
                  const std::optional<Legion::FutureMap>&) const
  {
    return false;
  }

  template <typename T, typename U, std::enable_if_t<!std::is_same_v<T, U>>* = nullptr>
  bool operator()(const T&, const U&) const
  {
    LEGATE_ABORT(
      "kind() and other.kind() are the same, but underlying variant types are different. Storage "
      "data is inconsistent with kind().");
  }
};

}  // namespace

bool Storage::equal(const Storage& other) const
{
  if (this == &other) {
    return true;
  }

  if (kind() != other.kind()) {
    return false;
  }

  return std::visit(EqualVisitor{}, storage_data_, other.storage_data_);
}

bool Storage::is_mapped() const
{
  if (unbound()) {
    return false;
  }

  // TODO(wonchanl): future- and future map-backed storages are considered unmapped until we
  // implement the full state machine for them (they are currently read only)
  return std::visit(Overload{
                      [&](const std::optional<InternalSharedPtr<LogicalRegionField>>& rf) {
                        return rf.has_value() && (*rf)->is_mapped();
                      },
                      [&](const std::optional<Legion::Future>&) { return false; },
                      [&](const std::optional<Legion::FutureMap>&) { return false; },
                    },
                    storage_data_);
}

namespace {

[[nodiscard]] bool can_tile_completely_for(Span<const std::uint64_t> shape,
                                           Span<const std::uint64_t> tile_shape,
                                           Span<const std::int64_t> offsets)

{
  const auto zipper           = zip_equal(shape, tile_shape, offsets);
  constexpr auto is_divisible = [](std::tuple<std::uint64_t, std::uint64_t, std::int64_t> tup) {
    const auto [shp, tshap, off] = tup;

    return ((shp % tshap) == 0) && ((off % tshap) == 0);
  };

  return std::all_of(zipper.begin(), zipper.end(), is_divisible) &&
         Runtime::get_runtime().partition_manager().use_complete_tiling(shape, tile_shape);
}

}  // namespace

InternalSharedPtr<Storage> Storage::slice(const InternalSharedPtr<Storage>& self,
                                          SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
                                          SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets)
{
  LEGATE_ASSERT(self.get() == this);

  if (Kind::FUTURE == kind()) {
    return self;
  }

  auto&& root                    = get_root(self);
  const auto& shape              = root->extents();
  const auto can_tile_completely = can_tile_completely_for(shape, tile_shape, offsets);

  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_shape, color;

  color_shape.reserve(shape.size());
  color.reserve(shape.size());
  if (can_tile_completely) {
    std::transform(shape.begin(),
                   shape.end(),
                   tile_shape.begin(),
                   std::back_inserter(color_shape),
                   std::divides<>{});
    std::transform(offsets.begin(),
                   offsets.end(),
                   tile_shape.begin(),
                   std::back_inserter(color),
                   std::divides<>{});
    std::fill(offsets.begin(), offsets.end(), 0);
  } else {
    color_shape.assign(tags::size_tag, shape.size(), 1);
    color.assign(tags::size_tag, shape.size(), 0);
  }

  auto tiling = create_tiling(std::move(tile_shape), std::move(color_shape), std::move(offsets));
  auto storage_partition = root->create_partition(root, std::move(tiling), can_tile_completely);

  return storage_partition->get_child_storage(storage_partition, std::move(color));
}

const Storage* Storage::get_root() const
{
  return parent_.has_value() ? (*parent_)->get_root() : this;
}

Storage* Storage::get_root() { return parent_.has_value() ? (*parent_)->get_root() : this; }

InternalSharedPtr<const Storage> Storage::get_root(
  const InternalSharedPtr<const Storage>& self) const
{
  LEGATE_ASSERT(self.get() == this);
  return parent_.has_value() ? (*parent_)->get_root(*parent_) : self;
}

InternalSharedPtr<Storage> Storage::get_root(const InternalSharedPtr<Storage>& self)
{
  LEGATE_ASSERT(self.get() == this);
  return parent_.has_value() ? (*parent_)->get_root(*parent_) : self;
}

const InternalSharedPtr<LogicalRegionField>& Storage::get_region_field() const
{
  auto&& rf = region_field_();

  LEGATE_CHECK(rf.has_value());
  return *rf;
}

Legion::Future Storage::get_future() const
{
  return std::visit(
    Overload{
      [](const std::optional<InternalSharedPtr<LogicalRegionField>>&) -> Legion::Future {
        LEGATE_ABORT("Cannot get future from RegionField-backed storage");
      },
      [](const std::optional<Legion::Future>& fut) { return fut.value_or(Legion::Future{}); },
      [](const std::optional<Legion::FutureMap>& fm) {
        LEGATE_CHECK(fm.has_value());
        // this future map must always exist, otherwise something bad has happened
        return fm->get_future(fm->get_future_map_domain().lo());
      },
    },
    storage_data_);
}

Legion::FutureMap Storage::get_future_map() const
{
  auto&& fm = future_map_();

  LEGATE_CHECK(fm.has_value());
  return *fm;
}

std::variant<Legion::Future, Legion::FutureMap> Storage::get_future_or_future_map(
  const Domain& launch_domain) const
{
  return std::visit(
    Overload{
      [](const std::optional<InternalSharedPtr<LogicalRegionField>>&)
        -> std::variant<Legion::Future, Legion::FutureMap> {
        LEGATE_ABORT("Cannot get future from RegionField-backed storage");
      },
      [](const std::optional<Legion::Future>&) -> std::variant<Legion::Future, Legion::FutureMap> {
        // TODO(jfaibussowit):
        // This seems... patently false. But the previous code asserted that kind() ==
        // FUTURE_MAP
        LEGATE_ABORT("Cannot get future from Future-backed storage");
      },
      [&](const std::optional<Legion::FutureMap>& fm_opt)
        -> std::variant<Legion::Future, Legion::FutureMap> {
        const auto& future_map       = fm_opt.value();
        const auto future_map_domain = future_map.get_future_map_domain();

        if (!launch_domain.is_valid()) {
          return future_map.get_future(future_map_domain.lo());
        }
        if (launch_domain == future_map_domain) {
          return future_map;
        }
        if (launch_domain.get_volume() != future_map_domain.get_volume()) {
          return future_map.get_future(future_map_domain.lo());
        }
        return Runtime::get_runtime().reshape_future_map(future_map, launch_domain);
      },
    },
    storage_data_);
}

void Storage::set_region_field(InternalSharedPtr<LogicalRegionField>&& region_field)
{
  std::visit(Overload{
               [&](std::optional<InternalSharedPtr<LogicalRegionField>>& storage_rf) {
                 LEGATE_CHECK(unbound_ && !region_field_().has_value());
                 LEGATE_CHECK(!parent_.has_value());

                 unbound_   = false;
                 storage_rf = std::move(region_field);
                 if (destroyed_out_of_order_) {
                   (*storage_rf)->allow_out_of_order_destruction();
                 }
                 Runtime::get_runtime().attach_alloc_info(*storage_rf, provenance());
               },
               [](const std::optional<Legion::Future>&) {
                 LEGATE_ABORT("Cannot set a region field on a future-backed storage.");
               },
               [](const std::optional<Legion::FutureMap>&) {
                 LEGATE_ABORT("Cannot set a region field on a future map-backed storage.");
               },
             },
             storage_data_);
}

void Storage::set_future(Legion::Future future, std::size_t scalar_offset)
{
  std::visit(Overload{
               [](const std::optional<InternalSharedPtr<LogicalRegionField>>&) {
                 LEGATE_ABORT("Cannot set a future on a region field-backed storage.");
               },
               [&](std::optional<Legion::Future>& storage_future) {
                 scalar_offset_ = scalar_offset;
                 storage_future = std::move(future);
               },
               [&](std::optional<Legion::FutureMap>&) {
                 // If we're here, that means that this was a replicated future that gets updated
                 // via reductions, so we reset the stale future map and update the kind
                 // TODO(wonchanl): true future map-backed stores aren't exposed to the user yet
                 // so if it wasn't replicated, something bad must have happened
                 LEGATE_CHECK(replicated_);
                 replicated_    = false;
                 scalar_offset_ = scalar_offset;
                 storage_data_.emplace<std::optional<Legion::Future>>(std::move(future));
               },
             },
             storage_data_);
}

void Storage::set_future_map(Legion::FutureMap future_map, std::size_t scalar_offset)
{
  std::visit(Overload{
               [](const std::optional<InternalSharedPtr<LogicalRegionField>>&) {
                 LEGATE_ABORT("Cannot set a future map on a region field-backed storage.");
               },
               [&](const std::optional<Legion::Future>&) {
                 // If this was originally a future-backed storage, it means this storage is now
                 // backed by a future map with futures having the same value
                 replicated_    = true;
                 scalar_offset_ = scalar_offset;
                 storage_data_.emplace<std::optional<Legion::FutureMap>>(std::move(future_map));
               },
               [&](std::optional<Legion::FutureMap>& storage_fm) {
                 scalar_offset_ = scalar_offset;
                 storage_fm     = std::move(future_map);
               },
             },
             storage_data_);
}

// Mapping is a logically non-const operation
RegionField Storage::map(  // NOLINT(readability-make-member-function-const)
  legate::mapping::StoreTarget target)
{
  return std::visit(Overload{
                      [&](std::optional<InternalSharedPtr<LogicalRegionField>>& rf) {
                        LEGATE_CHECK(rf.has_value());

                        auto mapped = (*rf)->map(target);
                        // Set the right subregion so the physical store can see the right domain
                        mapped.set_logical_region((*rf)->region());
                        return mapped;
                      },
                      [](const std::optional<Legion::Future>&) -> RegionField {
                        LEGATE_ABORT("Cannot map a future-backed storage.");
                      },
                      [](const std::optional<Legion::FutureMap>&) -> RegionField {
                        LEGATE_ABORT("Cannot map a future-map-backed storage.");
                      },
                    },
                    storage_data_);
}

// Unmapping is a logically non-const operation
void Storage::unmap()  // NOLINT(readability-make-member-function-const)
{
  std::visit(Overload{
               [&](std::optional<InternalSharedPtr<LogicalRegionField>>& rf) {
                 if (rf.has_value()) {
                   (*rf)->unmap();
                 }
               },
               [](const std::optional<Legion::Future>&) {},
               [](const std::optional<Legion::FutureMap>&) {},
             },
             storage_data_);
}

void Storage::allow_out_of_order_destruction()
{
  // Technically speaking this property only needs to be tracked on (root) LogicalRegionFields, but
  // a Storage may not have instantiated its region_field_ yet, so we note this also on the (root)
  // Storage, in case we need to propagate later. We only need to note this on the root Storage,
  // because any call that sets region_field_ (get_region_field(), set_region_field()) will end up
  // touching the root Storage.
  if (parent_.has_value()) {
    get_root()->allow_out_of_order_destruction();
  } else if (!destroyed_out_of_order_) {
    destroyed_out_of_order_ = true;
    std::visit(Overload{
                 [&](std::optional<InternalSharedPtr<LogicalRegionField>>& rf) {
                   if (rf.has_value()) {
                     (*rf)->allow_out_of_order_destruction();
                   }
                 },
                 [](const std::optional<Legion::Future>&) {},
                 [](const std::optional<Legion::FutureMap>&) {},
               },
               storage_data_);
  }
}

void Storage::free_early()
{
  std::visit(Overload{
               [&](std::optional<InternalSharedPtr<LogicalRegionField>>&) {
                 if (!unbound()) {
                   get_region_field()->release_region_field();
                 }
               },
               [&](std::optional<Legion::Future>&) {
                 // Do nothing
               },
               [&](std::optional<Legion::FutureMap>&) {
                 // Do nothing
               },
             },
             storage_data_);
}

Restrictions Storage::compute_restrictions() const { return Restrictions{dim()}; }

std::optional<InternalSharedPtr<Partition>> Storage::find_key_partition(
  const mapping::detail::Machine& machine,
  const ParallelPolicy& parallel_policy,
  const Restrictions& restrictions) const
{
  const auto new_num_pieces = machine.count() * parallel_policy.overdecompose_factor();

  if ((num_pieces_ == new_num_pieces) && key_partition_.has_value() &&
      restrictions.are_satisfied_by(**key_partition_)) {
    return key_partition_;
  }
  if (parent_.has_value()) {
    return (*parent_)->find_key_partition(machine, parallel_policy, restrictions);
  }
  return std::nullopt;
}

void Storage::set_key_partition(const mapping::detail::Machine& machine,
                                InternalSharedPtr<Partition> key_partition)
{
  num_pieces_    = machine.count();
  key_partition_ = std::move(key_partition);
}

void Storage::reset_key_partition() noexcept { key_partition_.reset(); }

InternalSharedPtr<StoragePartition> Storage::create_partition(
  const InternalSharedPtr<Storage>& self,
  InternalSharedPtr<Partition> partition,
  std::optional<bool> complete)
{
  LEGATE_ASSERT(self.get() == this);
  if (!complete.has_value()) {
    complete = partition->is_complete_for(*this);
  }
  return make_internal_shared<StoragePartition>(self, std::move(partition), *complete);
}

std::string Storage::to_string() const
{
  const auto kind_str = std::visit(
    Overload{
      [&](const std::optional<InternalSharedPtr<LogicalRegionField>>&) -> std::string_view {
        return "Region";
      },
      [&](const std::optional<Legion::Future>&) -> std::string_view { return "Future"; },
      [&](const std::optional<Legion::FutureMap>&) -> std::string_view { return "Future map"; },
    },
    storage_data_);

  auto result = fmt::format("Storage({}) {{kind: {}, level: {}", id(), kind_str, level());

  std::visit(
    Overload{
      [&](const std::optional<InternalSharedPtr<LogicalRegionField>>& rf) {
        if (unbound()) {
          result += ", region: unbound";
        } else if (rf.has_value()) {
          fmt::format_to(
            std::back_inserter(result), ", region: {}, field: {}", **rf, (*rf)->field_id());
        }
      },
      [](const std::optional<Legion::Future>&) {},
      [](const std::optional<Legion::FutureMap>&) {},
    },
    storage_data_);
  result += '}';
  return result;
}

InternalSharedPtr<Storage> slice_storage(const InternalSharedPtr<Storage>& self,
                                         SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
                                         SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets)
{
  return self->slice(self, std::move(tile_shape), std::move(offsets));
}

InternalSharedPtr<StoragePartition> create_storage_partition(const InternalSharedPtr<Storage>& self,
                                                             InternalSharedPtr<Partition> partition,
                                                             std::optional<bool> complete)
{
  return self->create_partition(self, std::move(partition), std::move(complete));
}

}  // namespace legate::detail
