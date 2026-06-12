/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/inline_storage.h>
#include <legate/partitioning/detail/restriction.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace legate {

class ParallelPolicy;

}  // namespace legate

namespace legate::mapping::detail {

class Machine;

}  // namespace legate::mapping::detail

namespace legate::detail {

class Type;
class Shape;
class Partition;
class TransformStack;
class ExternalAllocation;

class Storage {
 public:
  /**
   * @brief Construct a Storage object lazily.
   *
   * The backing of the storage could be backed by a region field
   * or a future/future map depending on the shape and whether
   * optimizing for a scalar. If executing with inline task launches,
   * the storage is an inline storage.
   *
   * @param shape The shape of the storage.
   * @param field_size The size of the field.
   * @param provenance The provenance of the storage.
   */
  Storage(InternalSharedPtr<Shape> shape, std::uint32_t field_size, std::string_view provenance);
  /**
   * @brief Construct a Storage object initialized eagerly from an existing scalar.
   *
   * The backing of the storage is a future unless executing with inline task launches,
   * in which case it is an inline storage. The scalar data is copied into the created storage.
   *
   * @param shape The shape of the storage.
   * @param scalar The scalar to copy into the storage.
   * @param provenance The provenance of the storage.
   */
  Storage(InternalSharedPtr<Shape> shape, const Scalar& scalar, std::string_view provenance);
  /**
   * @brief Construct a Storage object from an existing external allocation.
   *
   * This constructor is only valid when executing with inline task launches.
   *
   * Changes to the data underlying this storage will be reflected in the external allocation.
   *
   * @param shape The shape of the storage.
   * @param type The type of the storage.
   * @param allocation The external allocation that holds the pointer backing the storage.
   * @param provenance The provenance of the storage.
   */
  Storage(InternalSharedPtr<Shape> shape,
          const InternalSharedPtr<Type>& type,
          InternalSharedPtr<ExternalAllocation> allocation,
          std::string_view provenance);

  Storage(Storage&&) noexcept            = default;
  Storage& operator=(Storage&&) noexcept = default;

  [[nodiscard]] std::uint64_t id() const;
  [[nodiscard]] bool unbound() const;
  [[nodiscard]] bool deferred_bound() const;
  [[nodiscard]] const InternalSharedPtr<Shape>& shape() const;
  [[nodiscard]] Span<const std::uint64_t> extents() const;
  [[nodiscard]] std::size_t volume() const;
  [[nodiscard]] std::uint32_t dim() const;
  [[nodiscard]] bool overlaps(const InternalSharedPtr<Storage>& other) const;
  /**
   * @brief Determine whether two stores refer to the same memory.
   *
   * This routine can be used to determine whether two seemingly unrelated stores refer to the
   * same logical memory region.
   *
   * @param other The `Storage` to compare with.
   *
   * @return `true` if two stores cover the same underlying memory region, `false` otherwise.
   */
  [[nodiscard]] bool equal(const Storage& other) const;
  [[nodiscard]] std::string_view provenance() const;
  [[nodiscard]] bool is_mapped() const;
  [[nodiscard]] bool has_scalar_storage() const;

  [[nodiscard]] InternalSharedPtr<Storage> slice(
    const InternalSharedPtr<Storage>& self,
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
    SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets);

  void set_bound(bool bound);

  [[nodiscard]] InternalSharedPtr<InlineStorage> map(const Domain& domain,
                                                     legate::mapping::StoreTarget target);

  void unmap();
  void allow_out_of_order_destruction();
  void free_early();

  [[nodiscard]] Restrictions compute_restrictions() const;
  [[nodiscard]] std::optional<InternalSharedPtr<Partition>> find_key_partition(
    const mapping::detail::Machine& machine,
    const ParallelPolicy& parallel_policy,
    const Restrictions& restrictions) const;
  void set_key_partition(const mapping::detail::Machine& machine,
                         InternalSharedPtr<Partition> key_partition);
  void reset_key_partition() noexcept;

  [[nodiscard]] std::string to_string() const;

 private:
  std::uint64_t storage_id_{};
  bool unbound_{};
  InternalSharedPtr<Shape> shape_{};
  std::uint32_t field_size_{};
  std::string_view provenance_{};  // only relevant on the root Storage

  InternalSharedPtr<InlineStorage> storage_data_{};

  std::uint32_t num_pieces_{};
  std::optional<InternalSharedPtr<Partition>> key_partition_{};
};

}  // namespace legate::detail

#include <legate/data/detail/storage.inl>
