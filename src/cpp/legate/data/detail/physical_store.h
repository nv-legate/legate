/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/buffer.h>
#include <legate/geometry/point.h>
#include <legate/geometry/rect.h>
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
 * by folding the originating store's transform stack into the backing `Buffer`'s layout. A
 * `PhysicalStore` keeps this directly and no longer carries a transform stack.
 */
struct PhysicalStoreLayout {
  // Address of the store's coordinate origin (the point (0, ..., 0)). Accessors index absolutely
  // from here following `addr(p) = base + field_size * dot(p, strides)`.
  void* base{};
  // Per-dimension element strides (not bytes) over the store's (transformed) dimensions.
  Point strides{};
  // The store's (transformed) domain.
  Rect domain{};
};

class PhysicalStore {
 public:
  /**
   * @brief Constructs a bound physical store on top of a backing `Buffer`.
   *
   * The store shares ownership of @p buffer with the storage it was mapped from, so the buffer
   * (and its raw allocation) stays alive for as long as this physical store is alive. The buffer
   * is the single home for the metadata (raw allocation, element type, and storage domain) shared
   * between the storage and this physical store; @p layout carries the store-space affine view
   * derived from it (and the originating store's transform).
   *
   * @param permission The access the store grants (read-write or reduction-only).
   * @param buffer The buffer backing this store.
   * @param layout The store's (transformed) affine layout.
   */
  PhysicalStore(StorePermission permission,
                InternalSharedPtr<Buffer> buffer,
                PhysicalStoreLayout layout);

  /**
   * @brief Constructs a physical store for an unbound output store.
   *
   * The store has no backing allocation yet: the task body is expected to produce the data
   * by calling `create_output_buffer(..., bind_buffer=true)` or `bind_data(...)`, which allocates
   * a `RawAllocation` on @p target and binds it back into @p originating_store.
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
   * @brief Returns the store's domain as a NDRect<DIM>
   *
   * @return Store's domain as a NDRect<DIM>
   */
  [[nodiscard]] const Rect& domain() const;

  /**
   * @return The buffer backing this store, or null for an unbound store that has not been bound to
   * a buffer yet. The buffer's raw allocation (its base pointer, size, and memory kind) is reached
   * indirectly through `Buffer::get_raw_allocation()`.
   */
  [[nodiscard]] const InternalSharedPtr<Buffer>& get_buffer() const;
  [[nodiscard]] mapping::StoreTarget target() const;
  [[nodiscard]] bool is_accessible(mapping::TaskTarget task_target) const;

  [[nodiscard]] bool is_readable() const;
  [[nodiscard]] bool is_writable() const;
  [[nodiscard]] bool is_reducible() const;
  [[nodiscard]] bool is_unbound_store() const;

  [[nodiscard]] InternalSharedPtr<Buffer> create_output_buffer(const Point& extents,
                                                               bool bind_buffer);

  void bind_data(InternalSharedPtr<Buffer> buffer, const Point& extents);

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
  [[nodiscard]] const Point& strides() const;

 private:
  // The element type of an unbound store, needed before a backing buffer exists. Once the store is
  // bound (or was constructed bound), the type is read from `buffer_` instead (see `type()`).
  InternalSharedPtr<Type> type_{};

  std::int32_t dim_{-1};

  bool readable_{};
  bool writable_{};
  bool reducible_{};

  // The buffer that holds the manually-managed memory of the store, bundling the raw allocation,
  // element type, and storage domain that the store shares with its originating storage. Null for
  // an unbound store that has not been bound to a buffer yet. The store shares ownership of the
  // buffer, so the memory stays alive for as long as this physical store is alive.
  InternalSharedPtr<Buffer> buffer_{};

  // The store's affine layout in its own (transformed) coordinate space, computed at
  // store-creation time by folding the originating store's transform into the buffer's layout (see
  // `PhysicalStoreLayout`). `base_` is the address of the store's coordinate origin and `strides_`
  // are element strides over the store's dimensions.
  void* base_{};
  Point strides_{};

  // The store's (transformed) domain over its own coordinate space.
  Rect domain_{};

  // Unbound-store state. `unbound_` is true for an unbound store that has not been bound yet;
  // `unbound_target_` is the memory a produced buffer should be allocated on, and
  // `originating_store_` is the LogicalStore that gets updated in place when the task binds data.
  bool unbound_{false};
  mapping::StoreTarget unbound_target_{};
  InternalSharedPtr<LogicalStore> originating_store_{};
};

}  // namespace legate::detail

#include <legate/data/detail/physical_store.inl>
