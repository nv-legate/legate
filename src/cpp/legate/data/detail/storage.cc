/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/storage.h>

#include <legate_defines.h>

#include <legate/data/detail/buffer.h>
#include <legate/data/detail/copy_request.h>
#include <legate/data/detail/data_movement_requests.h>
#include <legate/data/detail/external_allocation.h>
#include <legate/data/detail/reduction_request.h>
#include <legate/data/detail/shape.h>
#include <legate/data/external_allocation.h>
#include <legate/mapping/detail/machine.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partition/no_partition.h>
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
    provenance_{std::move(provenance)}
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

  set_key_partition(create_no_partition());

  const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};

  auto&& api = cuda::detail::get_cuda_driver_api();

  api->mem_cpy_async(
    storage_data_->data(), scalar.data(), scalar.size(), Runtime::get_runtime().get_cuda_stream());
}

Storage::Storage(InternalSharedPtr<Shape> shape,
                 const InternalSharedPtr<Type>& type,
                 InternalSharedPtr<ExternalAllocation> allocation,
                 std::string_view provenance)
  : storage_id_{Runtime::get_runtime().get_unique_storage_id()},
    shape_{std::move(shape)},
    provenance_{std::move(provenance)}
{
  storage_data_ =
    make_internal_shared<InlineStorage>(to_domain(this->shape()->extents()),
                                        type->size(),
                                        legate::ExternalAllocation{std::move(allocation)});

  set_key_partition(create_no_partition());
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

void Storage::set_bound(bool bound) { unbound_ = !bound; }

std::tuple<InternalSharedPtr<InlineStorage>, Domain, bool>
Storage::map(  // NOLINT(readability-make-member-function-const)
  const Domain& domain,
  const InternalSharedPtr<Partition>& partition,
  const DomainPoint& color,
  legate::mapping::StoreTarget target,
  bool is_read,
  DataMovementRequests* requests)
{
  LEGATE_CHECK(!unbound());

  // TODO(wonchanl): Better way to check invalid colors
  auto sub_domain = color.dim != 0 ? partition->shard(domain, color) : domain;

  // If no data needs to be moved to serve the requested partition, return the current storage
  // immediately
  //
  // TODO(wonchanl): Since NoPartition will always return true, the current policy will keep the
  // initial replicated storage even when the entire storage gets written with a disjoint partition
  // by a downstream writer. There should be a way for users to reset the distribution in this case.
  if (key_partition_.has_value() && (*key_partition_)->subsumes(domain, partition)) {
    LEGATE_ASSERT(storage_data_);

    // TODO(wonchanl): GPU-to-CPU copies must be temporary copy-outs, not permanent remapping
    storage_data_->remap_to(target, requests);

    LEGATE_ASSERT(sub_domain.dim == 0 || sub_domain.get_volume() == 0 ||
                  storage_data_->domain().intersection(sub_domain) == sub_domain);

    return {storage_data_, sub_domain, true};
  }

  // LEGATE_CHECK(partition->is_complete_for(domain));

  // Otherwise, we need to allocate a new storage
  auto data = make_internal_shared<InlineStorage>(sub_domain, field_size_, target);

  if (!storage_data_) {
    storage_data_ = data;

    return {data, sub_domain, false};
  }

  // TODO(wonchanl): Need to handle copy-outs
  LEGATE_CHECK(is_read);
  LEGATE_ASSERT(key_partition_);

  requests->register_copy_in(
    CopyRequest{storage_data_, data, *key_partition_, partition, domain, field_size_});

  return {data, sub_domain, false};
}

std::pair<InternalSharedPtr<InlineStorage>, Domain>
Storage::map_reduction(  // NOLINT(readability-make-member-function-const)
  const Domain& domain,
  const InternalSharedPtr<Partition>& partition,
  const DomainPoint& color,
  legate::mapping::StoreTarget target,
  bool is_parallel,
  const InternalSharedPtr<Type>& type,
  ReductionOpKind op_kind,
  DataMovementRequests* requests)
{
  LEGATE_ASSERT(!unbound());

  // TODO(wonchanl): Better way to check invalid colors
  auto sub_domain = color.dim != 0 ? partition->shard(domain, color) : domain;

  // No subsumption checks for reductions as we need to create a new storage anyway
  //
  // TODO(wonchanl): Put back the optimization that converts reductions to read-write accesses when
  // the partition is disjoint

  if (!is_parallel && key_partition_.has_value() && !(*key_partition_)->has_color_shape()) {
    return {storage_data_, sub_domain};
  }

  // TODO(wonchanl): For now we create a full-size reduction buffer when the target is replicated.
  // If we have proper tile-wise reduction support, we should pass the `sub_domain` as the storage
  // domain.
  auto tmp = make_internal_shared<InlineStorage>(domain, field_size_, target);

  // Initialize the per-task scratch with the reduction identity so that ranks that don't write
  // to a given cell don't perturb the final reduction result.
  fill_with_reduction_identity(tmp->data(), domain.get_volume(), target, type->code, op_kind);

  // TODO(wonchanl): We need to raise an uninitialized access exception if these don't hold
  LEGATE_CHECK(storage_data_);
  LEGATE_CHECK(key_partition_.has_value());

  requests->register_reduction(
    ReductionRequest{tmp, storage_data_, partition, *key_partition_, domain, type, op_kind});

  return {std::move(tmp), sub_domain};
}

// Unmapping is a logically non-const operation
void Storage::unmap()  // NOLINT(readability-make-member-function-const)
{
}

void Storage::free_early()
{
  // TODO(wonchanl): Need a v2 implementation
}

Restrictions Storage::compute_restrictions() const { return Restrictions{dim()}; }

std::optional<InternalSharedPtr<Partition>> Storage::find_key_partition(
  const Restrictions& restrictions) const
{
  if (key_partition_.has_value() && restrictions.are_satisfied_by(**key_partition_, shape())) {
    return key_partition_;
  }
  return std::nullopt;
}

void Storage::set_key_partition(InternalSharedPtr<Partition> key_partition)
{
  key_partition_ = std::move(key_partition);
}

void Storage::reset_key_partition() noexcept { key_partition_.reset(); }

std::string Storage::to_string() const { return fmt::format("Storage(id: {})", id()); }

}  // namespace legate::detail
