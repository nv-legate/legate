/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/storage.h>

#include <legate_defines.h>

#include <legate/data/detail/buffer.h>
#include <legate/data/detail/external_allocation.h>
#include <legate/data/detail/future_wrapper.h>
#include <legate/data/detail/physical_stores/future_physical_store.h>
#include <legate/data/detail/physical_stores/inline_physical_store.h>
#include <legate/data/detail/physical_stores/region_physical_store.h>
#include <legate/data/detail/shape.h>
#include <legate/data/detail/storage_partition.h>
#include <legate/data/external_allocation.h>
#include <legate/mapping/detail/machine.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/tuning/parallel_policy.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/legion_utilities.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/dispatch.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace legate::detail {

Storage::Storage(InternalSharedPtr<Shape> shape,
                 std::uint32_t field_size,
                 std::string_view provenance)
  : storage_id_{Runtime::get_runtime().get_unique_storage_id()},
    unbound_{shape->unbound()},
    shape_{std::move(shape)},
    field_size_{field_size},
    provenance_{std::move(provenance)},
    offsets_{tags::size_tag, dim(), 0}
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    log_legate().debug() << "Create " << to_string();
  }
}

Storage::Storage(InternalSharedPtr<Shape> shape, const Scalar& scalar, std::string_view provenance)
  : Storage{shape, static_cast<std::uint32_t>(scalar.size()), provenance}
{
  // No alignment guarantee for the pointer provided by Scalar,
  // so we need to create a new allocation and copy the data over.
  const Domain domain = to_domain(this->shape()->extents());

  storage_data_ =
    make_internal_shared<InlineStorage>(domain, field_size_, mapping::StoreTarget::SYSMEM);

  std::memcpy(storage_data_->data(), scalar.data(), scalar.size());
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
    storage_data_{}
{
  // TODO(wonchanl): Need a v2 implementation
  LEGATE_ABORT("Sub-storage has not been implemented");
}

Storage::Storage(InternalSharedPtr<Shape> shape,
                 const InternalSharedPtr<Type>& type,
                 InternalSharedPtr<ExternalAllocation> allocation,
                 std::string_view provenance)
  : storage_id_{Runtime::get_runtime().get_unique_storage_id()},
    shape_{std::move(shape)},
    provenance_{std::move(provenance)},
    offsets_{tags::size_tag, dim(), 0}
{
  storage_data_ =
    make_internal_shared<InlineStorage>(to_domain(this->shape()->extents()),
                                        type->size(),
                                        legate::ExternalAllocation{std::move(allocation)});
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

bool Storage::equal(const Storage& other) const
{
  return (this == &other) || (storage_data_ == other.storage_data_);
}

bool Storage::is_mapped() const
{
  if (unbound()) {
    return false;
  }

  // TODO(wonchanl): Need a v2 implementation
  return false;
}

bool Storage::has_scalar_storage() const { return false; }

InternalSharedPtr<Storage> Storage::slice(const InternalSharedPtr<Storage>& /*self*/,
                                          SmallVector<std::uint64_t, LEGATE_MAX_DIM> /*tile_shape*/,
                                          SmallVector<std::int64_t, LEGATE_MAX_DIM> /*offsets*/)
{
  // TODO(wonchanl): Need a v2 implementation
  return nullptr;
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

void Storage::set_bound(bool bound) { unbound_ = !bound; }

// Mapping is a logically non-const operation
InternalSharedPtr<PhysicalStore> Storage::map(  // NOLINT(readability-make-member-function-const)
  legate::mapping::StoreTarget /*target*/,
  InternalSharedPtr<Type> type,
  const InternalSharedPtr<Shape>& shape,
  InternalSharedPtr<TransformStack> transform,
  bool /*ignore_future_mutability*/,
  const std::optional<Domain>& domain)
{
  if (unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound storages cannot be inlined mapped"};
  }

  auto root = get_root();

  LEGATE_CHECK(root == this);

  return make_internal_shared<InlinePhysicalStore>(shape->ndim(),
                                                   std::move(type),
                                                   GlobalRedopID{-1},
                                                   transform,
                                                   LEGION_READ_WRITE,
                                                   storage_data_,
                                                   *domain);
}

InternalSharedPtr<PhysicalStore> Storage::map(  // NOLINT(readability-make-member-function-const)
  legate::mapping::StoreTarget target,
  InternalSharedPtr<Type> type,
  const InternalSharedPtr<Shape>& shape,
  InternalSharedPtr<TransformStack> transform,
  const Domain& domain)
{
  if (unbound()) {
    throw TracedException<std::invalid_argument>{"Unbound storages cannot be inlined mapped"};
  }

  auto root = get_root();

  LEGATE_CHECK(root == this);

  if (!storage_data_) {
    storage_data_ = make_internal_shared<InlineStorage>(domain, field_size_, target);
  }

  storage_data_->remap_to(target);

  return make_internal_shared<InlinePhysicalStore>(shape->ndim(),
                                                   std::move(type),
                                                   GlobalRedopID{-1},
                                                   transform,
                                                   LEGION_READ_WRITE,
                                                   storage_data_,
                                                   domain);
}

// Unmapping is a logically non-const operation
void Storage::unmap()  // NOLINT(readability-make-member-function-const)
{
}

void Storage::allow_out_of_order_destruction() {}

void Storage::free_early()
{
  // TODO(wonchanl): Need a v2 implementation
}

Restrictions Storage::compute_restrictions() const { return Restrictions{dim()}; }

std::optional<InternalSharedPtr<Partition>> Storage::find_key_partition(
  const mapping::detail::Machine& machine,
  const ParallelPolicy& parallel_policy,
  const Restrictions& restrictions) const
{
  const auto new_num_pieces = machine.count() * parallel_policy.overdecompose_factor();

  if ((num_pieces_ == new_num_pieces) && key_partition_.has_value() &&
      restrictions.are_satisfied_by(**key_partition_, shape())) {
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
  return fmt::format("Storage(id: {}, level: {})", id(), level());
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
