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
#include <legate/data/detail/remap_request.h>
#include <legate/data/detail/shape.h>
#include <legate/data/external_allocation.h>
#include <legate/mapping/detail/machine.h>
#include <legate/partitioning/detail/partition.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/tuning/parallel_policy.h>
#include <legate/utilities/detail/formatters.h>
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
    shape_{std::move(shape)},
    domain_{unbound() ? Rect{} : shape_->domain()},
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
  const auto target = Runtime::get_runtime().get_machine().preferred_scalar_store_target();
  storage_data_     = make_internal_shared<InlineStorage>(domain_, field_size_, target);

  set_key_partition(create_no_partition());

  if (target == mapping::StoreTarget::SYSMEM) {
    std::memcpy(storage_data_->data(), scalar.data(), scalar.size());
    return;
  }

  const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};

  auto&& api = cuda::detail::get_cuda_driver_api();

  api->mem_cpy_async(
    storage_data_->data(), scalar.data(), scalar.size(), Runtime::get_runtime().get_cuda_stream());
}

Storage::Storage(InternalSharedPtr<Shape> shape,
                 std::uint32_t field_size,
                 InternalSharedPtr<ExternalAllocation> allocation,
                 std::string_view provenance)
  : Storage{shape, field_size, provenance}
{
  storage_data_ = make_internal_shared<InlineStorage>(
    domain_, field_size, legate::ExternalAllocation{std::move(allocation)});

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

std::tuple<InternalSharedPtr<InlineStorage>, Rect, bool>
Storage::map(  // NOLINT(readability-make-member-function-const)
  const Rect& domain,
  const InternalSharedPtr<Partition>& partition,
  const Point& color,
  legate::mapping::StoreTarget target,
  bool is_read,
  bool is_transformed,
  DataMovementRequests* requests)
{
  LEGATE_CHECK(!unbound());

  // TODO(wonchanl): Better way to check invalid colors
  auto sub_domain = color.dim() != 0 ? partition->shard(domain, color) : domain;

  // The canonical ("active") storage stays put on its own memory (by default the rank's GPU). When
  // the task runs on a different memory, it operates on a redundant copy on `target` instead, and
  // the canonical storage is left in place. `resolve_active` lazily produces the storage the task
  // operates on: the canonical storage itself when it already lives on `target`, otherwise a
  // redundant copy that is populated from the canonical storage by a remap copy-in. The copy is
  // shared across the input and output mappings of the same store (via `working_copies_`), so a
  // single remap copy-in populates it. `needs_remap` is false before the first materialization.
  const bool needs_remap = storage_data_ && storage_data_->target() != target;
  InternalSharedPtr<InlineStorage> active_copy;  // memoizes resolve_active within this call

  const auto resolve_active = [&]() -> const InternalSharedPtr<InlineStorage>& {
    if (active_copy) {
      return active_copy;
    }
    if (!needs_remap) {
      active_copy = storage_data_;
      return active_copy;
    }
    if (auto cached = requests->find_working_copy(id()); cached.has_value()) {
      active_copy = *std::move(cached);
      return active_copy;
    }
    active_copy = make_internal_shared<InlineStorage>(storage_data_->domain(), field_size_, target);
    requests->record_working_copy(id(), active_copy);

    auto copy_in = RemapRequest{storage_data_, active_copy, field_size_};

    // A device-to-host copy must be awaited before the (CPU) task reads the redundant copy.
    if (copy_in.requires_host_sync()) {
      requests->require_data_sync();
    }
    requests->register_remap_copy_in(std::move(copy_in));
    return active_copy;
  };

  // When the task writes through a redundant copy, the copy must be merged back into the canonical
  // storage after the task via a remap copy-out (ordered after any repartitioning copy-out).
  const auto register_remap_copy_out = [&] {
    if (needs_remap) {
      requests->register_remap_copy_out(RemapRequest{resolve_active(), storage_data_, field_size_});
    }
  };

  // If no repartitioning is needed, the task operates directly on the active storage (or its
  // redundant copy on `target`).
  //
  // TODO(wonchanl): Since NoPartition will always return true, the current policy will keep the
  // initial replicated storage even when the entire storage gets written with a disjoint partition
  // by a downstream writer. There should be a way for users to reset the distribution in this case.
  if (key_partition_.has_value() && (*key_partition_)->subsumes(domain, partition)) {
    LEGATE_ASSERT(storage_data_);

    auto&& active = resolve_active();

    if (!is_read) {
      register_remap_copy_out();
    }

    LEGATE_ASSERT(sub_domain.dim() == 0 || sub_domain.volume() == 0 ||
                  active->domain().intersection(sub_domain) == sub_domain);

    return {active, sub_domain, true};
  }

  // Registers a copy-out that merges the sub-region written into `data` back into the active
  // storage after the task runs (repartitioning it into the active storage's partition). Only
  // needed for transformed stores (e.g. slices) that are views into a larger storage; a
  // non-transformed write covers the whole store and simply replaces it. When the active storage is
  // a redundant copy, a remap copy-out then merges it back into the canonical storage.
  const auto register_copy_out = [&](const InternalSharedPtr<InlineStorage>& data) {
    LEGATE_ASSERT(storage_data_);
    LEGATE_ASSERT(key_partition_.has_value());
    requests->register_copy_out(CopyRequest{
      id(), data, resolve_active(), partition, *key_partition_, domain, domain_, field_size_});
    register_remap_copy_out();
  };

  // Finalizes a write into `data`: a transformed store (e.g. a slice) is a view into a larger
  // storage whose written sub-region is merged back via a copy-out, whereas a non-transformed
  // write covers the whole store and so its buffer becomes the new canonical storage.
  const auto finalize_write = [&](const InternalSharedPtr<InlineStorage>& data) {
    if (is_transformed) {
      register_copy_out(data);
    } else {
      storage_data_ = data;
    }
  };

  // Reuse an allocation already handed out for this storage during the current task launch. This
  // makes a store that is passed as both an input and an output (or accessed multiple times with
  // the same partition) map to a single physical allocation, so that the task's updates are seen
  // in place rather than being lost in a second, independent allocation.
  if (auto cached = requests->find_mapped_alloc(id());
      cached.has_value() && (*cached)->domain() == sub_domain) {
    auto data = *std::move(cached);

    if (!is_read) {
      finalize_write(data);
    }
    return {std::move(data), sub_domain, false};
  }

  // Otherwise, we need to allocate a new storage
  auto data = make_internal_shared<InlineStorage>(sub_domain, field_size_, target);

  requests->record_mapped_alloc(id(), data);

  if (!storage_data_) {
    // First materialization of the active storage: it lives on the memory of the task that first
    // writes it (by default the rank's GPU, since tasks run on the GPU when one is available). A
    // store first written on the GPU therefore keeps its active storage on the GPU; a task that
    // later runs on a different memory operates on a temporary redundant copy (see above) rather
    // than permanently migrating the active storage.
    storage_data_ = data;

    return {data, sub_domain, false};
  }

  LEGATE_ASSERT(key_partition_.has_value());

  if (is_read) {
    requests->register_copy_in(CopyRequest{
      id(), resolve_active(), data, *key_partition_, partition, domain_, domain, field_size_});
  } else {
    finalize_write(data);
  }

  return {data, sub_domain, false};
}

std::pair<InternalSharedPtr<InlineStorage>, Rect>
Storage::map_reduction(  // NOLINT(readability-make-member-function-const)
  const Rect& domain,
  const InternalSharedPtr<Partition>& partition,
  const Point& color,
  legate::mapping::StoreTarget target,
  bool is_parallel,
  const InternalSharedPtr<Type>& type,
  ReductionOpKind op_kind,
  bool store_has_identity,
  DataMovementRequests* requests)
{
  LEGATE_ASSERT(!unbound());

  // TODO(wonchanl): Better way to check invalid colors
  auto sub_domain = color.dim() != 0 ? partition->shard(domain, color) : domain;

  // Reducing directly into the canonical storage is only valid when it already lives on the memory
  // the reduction task runs on. Otherwise the task would reduce into a redundant copy on `target`
  // whose result is never merged back (`map_reduction` does not carry the remap copy-out machinery
  // that `map` does), silently dropping the reduction. When a remap is needed we fall through to
  // the "fresh buffer on `target` + ReductionRequest" path, which copies the result back into the
  // canonical storage.
  const bool needs_remap = storage_data_ && storage_data_->target() != target;

  // No subsumption checks for reductions as we need to create a new storage anyway
  //
  // TODO(wonchanl): Put back the optimization that converts reductions to read-write accesses when
  // the partition is disjoint

  if (!needs_remap && !is_parallel && key_partition_.has_value() &&
      !(*key_partition_)->has_color_shape()) {
    return {storage_data_, sub_domain};
  }

  // TODO(wonchanl): We need to raise an uninitialized access exception if these don't hold
  LEGATE_CHECK(storage_data_);
  LEGATE_CHECK(key_partition_.has_value());

  InternalSharedPtr<InlineStorage> reduction_buffer{};

  // If the store is just initialized with the reduction identity and the key partition subsumes the
  // partition used in reduction, we can use the storage directly as the reduction buffer (unless a
  // cross-memory remap is required; see above).
  if (!needs_remap && store_has_identity && (*key_partition_)->subsumes(domain, partition)) {
    reduction_buffer = storage_data_;

    // FIXME(wonchanl): We need a proper disjointedness check when the key_partition_ is a Tiling.
    if (!(*key_partition_)->has_color_shape()) {
      requests->register_reduction(ReductionRequest{
        reduction_buffer, storage_data_, partition, *key_partition_, domain, type, op_kind});
    }
  } else {
    // TODO(wonchanl): For now we create a full-size reduction buffer when the target is replicated.
    // If we have proper tile-wise reduction support, we should pass the `sub_domain` as the storage
    // domain.
    reduction_buffer = make_internal_shared<InlineStorage>(domain, field_size_, target);

    // Initialize the per-task scratch with the reduction identity so that ranks that don't write
    // to a given cell don't perturb the final reduction result.
    fill_with_reduction_identity(
      reduction_buffer->data(), domain.volume(), target, type->code, op_kind);

    requests->register_reduction(ReductionRequest{
      reduction_buffer, storage_data_, partition, *key_partition_, domain, type, op_kind});
  }

  return {std::move(reduction_buffer), sub_domain};
}

// Unmapping is a logically non-const operation
void Storage::unmap()  // NOLINT(readability-make-member-function-const)
{
}

void Storage::free_early()
{
  // TODO(wonchanl): Need a v2 implementation
}

mapping::StoreTarget Storage::target() const { return storage_data_->target(); }

void Storage::set_inline_storage(InternalSharedPtr<InlineStorage> storage_data)
{
  storage_data_ = std::move(storage_data);
}

void Storage::bind(InternalSharedPtr<InlineStorage> storage_data, const Rect& domain)
{
  LEGATE_CHECK(unbound());

  // A buffer bound to an unbound store must be zero-based: the store's extents are derived from the
  // buffer domain assuming its origin is (0, ..., 0).
  for (std::uint32_t i = 0; i < domain.dim(); ++i) {
    LEGATE_CHECK(domain.lo()[i] == 0);
  }

  storage_data_ = std::move(storage_data);
  domain_       = domain;
}

void Storage::finalize_unbound()
{
  shape_->bind_extents(domain_.extents());
  set_key_partition(create_no_partition());
}

void Storage::finalize_unbound_partitioned(Point global_extents,
                                           const Rect& local_global_domain,
                                           InternalSharedPtr<Partition> key_partition)
{
  LEGATE_CHECK(unbound());

  shape_->bind_extents(std::move(global_extents));
  domain_ = to_domain(shape_->extents());
  // Re-base this rank's local (zero-based) piece so its domain is the global sub-domain. The copy
  // engine assumes a storage's data at global coordinate `domain.lo()` maps to element 0, which is
  // exactly what re-basing produces here.
  storage_data_->rebase(local_global_domain);
  set_key_partition(std::move(key_partition));
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
