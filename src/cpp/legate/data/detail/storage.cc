/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/storage.h>

#include <legate_defines.h>

#include <legate/data/detail/buffer.h>
#include <legate/data/detail/external_allocation.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/shape.h>
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

  storage_data_ = make_internal_shared<InlineStorage>(
    domain, field_size_, Runtime::get_runtime().get_machine().preferred_scalar_store_target());

  std::memcpy(storage_data_->data(), scalar.data(), scalar.size());
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

  return lhs == rhs;
}

bool Storage::equal(const Storage& other) const { return this == &other; }

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

void Storage::set_bound(bool bound) { unbound_ = !bound; }

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

  if (!storage_data_) {
    storage_data_ = make_internal_shared<InlineStorage>(domain, field_size_, target);
  }

  storage_data_->remap_to(target);

  return make_internal_shared<PhysicalStore>(shape->ndim(),
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
  return std::nullopt;
}

void Storage::set_key_partition(const mapping::detail::Machine& machine,
                                InternalSharedPtr<Partition> key_partition)
{
  num_pieces_    = machine.count();
  key_partition_ = std::move(key_partition);
}

void Storage::reset_key_partition() noexcept { key_partition_.reset(); }

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

}  // namespace legate::detail
