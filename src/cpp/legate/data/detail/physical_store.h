/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/inline_storage.h>
#include <legate/data/inline_allocation.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <cstdint>

namespace legate::mapping {

enum class StoreTarget : std::uint8_t;

}

namespace legate::detail {

class LogicalStore;

/**
 * @brief The kind of access a mapped `PhysicalStore` grants through its accessors.
 *
 * A mapped store is either read-write (readable and writable) or reduction-only (reducible). This
 * mirrors the access modes the runtime maps stores with, and replaces the use of Legion privilege
 * modes for deciding which accessors a store permits.
 */
enum class StorePermission : std::uint8_t {
  READ_WRITE,  ///< The store may be read and written.
  REDUCE,      ///< The store may be reduced into.
};

/**
 * @brief The affine layout of a (possibly transformed) physical store.
 *
 * Computed once by the logic that creates a `PhysicalStore` (see `LogicalStore::map` and friends),
 * by folding the originating store's transform stack into the backing `InlineStorage`'s layout. A
 * `PhysicalStore` keeps this directly and no longer carries a transform stack.
 */
struct PhysicalStoreLayout {
  // Address of the store's coordinate origin (the point (0, ..., 0)). Accessors index absolutely
  // from here following `addr(p) = base + field_size * dot(p, strides)`.
  void* base{};
  // Per-dimension element strides (not bytes) over the store's (transformed) dimensions.
  DomainPoint strides{};
  // The store's (transformed) domain.
  Domain domain{};
};

class PhysicalStore {
 public:
  PhysicalStore(InternalSharedPtr<Type> type,
                StorePermission permission,
                InternalSharedPtr<InlineStorage> storage,
                PhysicalStoreLayout layout);

  /**
   * @brief Constructs a physical store for an unbound output store.
   *
   * The store has no backing `InlineStorage` yet: the task body is expected to produce the data
   * by calling `create_output_buffer(..., bind_buffer=true)` or `bind_data(...)`, which allocates
   * an `InlineStorage` on @p target and binds it back into @p originating_store.
   *
   * @param dim The (declared) dimensionality of the unbound store.
   * @param type The element type of the store.
   * @param target The memory kind the produced buffer should be allocated on (the store's mapping
   * decision).
   * @param originating_store The `LogicalStore` this physical store was mapped from, updated in
   * place when the task binds data.
   */
  PhysicalStore(std::int32_t dim,
                InternalSharedPtr<Type> type,
                mapping::StoreTarget target,
                InternalSharedPtr<LogicalStore> originating_store);

  PhysicalStore(PhysicalStore&& other) noexcept            = default;
  PhysicalStore& operator=(PhysicalStore&& other) noexcept = default;

  PhysicalStore(const PhysicalStore& other)            = delete;
  PhysicalStore& operator=(const PhysicalStore& other) = delete;

  ~PhysicalStore() = default;

  [[nodiscard]] bool valid() const;

  [[nodiscard]] std::int32_t dim() const;
  [[nodiscard]] const InternalSharedPtr<Type>& type() const;

  /**
   * @brief Returns the store's domain as a Rect<DIM>
   *
   * @return Store's domain as a Rect<DIM>
   */
  [[nodiscard]] const Domain& domain() const;
  [[nodiscard]] InlineAllocation get_inline_allocation() const;
  [[nodiscard]] mapping::StoreTarget target() const;
  [[nodiscard]] bool is_accessible(mapping::TaskTarget task_target) const;

  [[nodiscard]] bool is_readable() const;
  [[nodiscard]] bool is_writable() const;
  [[nodiscard]] bool is_reducible() const;
  [[nodiscard]] bool is_unbound_store() const;

  /**
   * @brief Allocates an `InlineStorage` for an output buffer of this (unbound) store.
   *
   * The storage spans @p bounds, holds elements of this store's type, and is allocated on the
   * store's mapped target memory. Used by `create_output_buffer`.
   *
   * @param bounds The domain the buffer should span.
   *
   * @return The freshly allocated inline storage.
   */
  [[nodiscard]] InternalSharedPtr<InlineStorage> create_output_buffer_storage(
    const Domain& bounds) const;

  /**
   * @brief Binds an inline storage (produced as a buffer by the task) to this unbound store.
   *
   * The originating `LogicalStore` is updated in place so downstream operations observe the
   * produced data, and this physical store is updated to reflect the now-bound data.
   *
   * @param storage The inline storage to bind.
   * @param domain The domain the store should span over @p storage.
   */
  void bind_data(InternalSharedPtr<InlineStorage> storage, const Domain& domain);

  /**
   * @brief Binds empty data to this unbound store (the store becomes a bound, zero-volume store).
   */
  void bind_empty_data();

  void check_shape_dimension(std::int32_t dim) const;
  void check_accessor_type(Type::Code code, std::size_t size_of_T) const;
  void check_accessor_dimension(std::int32_t dim) const;
  void check_write_access() const;
  void check_reduction_access() const;

  /**
   * @brief Returns the address of the store's coordinate origin (the point `(0, ..., 0)`).
   *
   * Accessors index absolutely from here following `addr(p) = base() + field_size * dot(p,
   * strides())`. Any store transformation (projection, shift, ...) was already folded into this
   * pointer when the store was created, so no further translation is needed.
   */
  [[nodiscard]] void* base() const;

  /**
   * @brief Returns the per-dimension element strides over the store's (transformed) dimensions.
   *
   * The strides are in element units (not bytes) and have `dim()` entries (at least one).
   */
  [[nodiscard]] const DomainPoint& strides() const;

 private:
  [[nodiscard]] const InternalSharedPtr<InlineStorage>& storage_() const;

  InternalSharedPtr<Type> type_{};

  std::int32_t dim_{-1};

  bool readable_{};
  bool writable_{};
  bool reducible_{};

  // The inline storage that holds the manually-managed memory of the store. Null for an unbound
  // store that has not been bound to a buffer yet.
  InternalSharedPtr<InlineStorage> inline_storage_{};

  // The store's affine layout in its own (transformed) coordinate space, computed at
  // store-creation time by folding the originating store's transform into `inline_storage_`'s
  // layout (see `PhysicalStoreLayout`). `base_` is the address of the store's coordinate origin
  // and `strides_` are element strides over the store's dimensions.
  void* base_{};
  DomainPoint strides_{};

  // The store's (transformed) domain over its own coordinate space.
  Domain domain_{};

  // Unbound-store state. `unbound_` is true for an unbound store that has not been bound yet;
  // `unbound_target_` is the memory a produced buffer should be allocated on, and
  // `originating_store_` is the LogicalStore that gets updated in place when the task binds data.
  bool unbound_{false};
  mapping::StoreTarget unbound_target_{};
  InternalSharedPtr<LogicalStore> originating_store_{};
};

}  // namespace legate::detail

#include <legate/data/detail/physical_store.inl>
