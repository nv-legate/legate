/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_region_field.h>
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

class LogicalStorePartition;
class PhysicalStore;
class StoragePartition;
class Strategy;
class TaskReturnLayoutForUnpack;
class Variable;
class Type;
class BufferBuilder;
class Shape;
class Partition;
class TransformStack;

class Storage {
 public:
  enum class Kind : std::uint8_t {
    REGION_FIELD,
    FUTURE,
    FUTURE_MAP,
  };

  // Create a RegionField-backed or a Future-backed storage.
  Storage(InternalSharedPtr<Shape> shape,
          std::uint32_t field_size,
          bool optimize_scalar,
          std::string_view provenance);
  // Create a Future-backed storage. Initialized eagerly.
  Storage(InternalSharedPtr<Shape> shape, Legion::Future future, std::string_view provenance);
  // Create a RegionField-backed sub-storage.
  Storage(SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents,
          InternalSharedPtr<StoragePartition> parent,
          SmallVector<std::uint64_t, LEGATE_MAX_DIM> color,
          SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets);
  ~Storage();

  Storage(Storage&&) noexcept            = default;
  Storage& operator=(Storage&&) noexcept = default;

  [[nodiscard]] std::uint64_t id() const;
  [[nodiscard]] bool replicated() const;
  [[nodiscard]] bool unbound() const;
  [[nodiscard]] const InternalSharedPtr<Shape>& shape() const;
  [[nodiscard]] Span<const std::uint64_t> extents() const;
  [[nodiscard]] Span<const std::int64_t> offsets() const;
  [[nodiscard]] std::size_t volume() const;
  [[nodiscard]] std::uint32_t dim() const;
  [[nodiscard]] bool overlaps(const InternalSharedPtr<Storage>& other) const;
  [[nodiscard]] Kind kind() const;
  [[nodiscard]] std::int32_t level() const;
  [[nodiscard]] std::size_t scalar_offset() const;
  [[nodiscard]] std::string_view provenance() const;
  [[nodiscard]] bool is_mapped() const;

  [[nodiscard]] InternalSharedPtr<Storage> slice(
    const InternalSharedPtr<Storage>& self,
    SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
    SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets);
  [[nodiscard]] const Storage* get_root() const;
  [[nodiscard]] Storage* get_root();
  [[nodiscard]] InternalSharedPtr<const Storage> get_root(
    const InternalSharedPtr<const Storage>& self) const;
  [[nodiscard]] InternalSharedPtr<Storage> get_root(const InternalSharedPtr<Storage>& self);

  [[nodiscard]] const InternalSharedPtr<LogicalRegionField>& get_region_field() const;
  [[nodiscard]] Legion::Future get_future() const;
  [[nodiscard]] Legion::FutureMap get_future_map() const;
  [[nodiscard]] std::variant<Legion::Future, Legion::FutureMap> get_future_or_future_map(
    const Domain& launch_domain) const;

  void set_region_field(InternalSharedPtr<LogicalRegionField>&& region_field);
  void set_future(Legion::Future future, std::size_t scalar_offset);
  void set_future_map(Legion::FutureMap future_map, std::size_t scalar_offset);

  [[nodiscard]] RegionField map(legate::mapping::StoreTarget target);
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

  [[nodiscard]] InternalSharedPtr<StoragePartition> create_partition(
    const InternalSharedPtr<Storage>& self,
    InternalSharedPtr<Partition> partition,
    std::optional<bool> complete = std::nullopt);

  [[nodiscard]] std::string to_string() const;

 private:
  // Private getters for member variables
  /**
   * @brief Return a const reference to the region field optional of this storage.
   *
   * @throw std::bad_variant_access if the underlying storage variant is not a region field.
   */
  [[nodiscard]] const std::optional<InternalSharedPtr<LogicalRegionField>>& region_field_() const;
  /**
   * @brief Return the region field optional of this storage.
   *
   * @throw std::bad_variant_access if the underlying storage variant is not a region field.
   */
  [[nodiscard]] std::optional<InternalSharedPtr<LogicalRegionField>>& region_field_();

  /**
   * @brief Return a const reference to the future optional of this storage.
   *
   * @throw std::bad_variant_access if the underlying storage variant is not a future.
   */
  [[nodiscard]] const std::optional<Legion::Future>& future_() const;
  /**
   * @brief Return the future optional of this storage.
   *
   * @throw std::bad_variant_access if the underlying storage variant is not a future.
   */
  [[nodiscard]] std::optional<Legion::Future>& future_();

  /**
   * @brief Return a const reference to the future map optional of this storage.
   *
   * @throw std::bad_variant_access if the underlying storage variant is not a future map.
   */
  [[nodiscard]] const std::optional<Legion::FutureMap>& future_map_() const;
  /**
   * @brief Return the future map optional of this storage.
   *
   * @throw std::bad_variant_access if the underlying storage variant is not a future map.
   */
  [[nodiscard]] std::optional<Legion::FutureMap>& future_map_();

  std::uint64_t storage_id_{};
  bool replicated_{};
  bool unbound_{};
  bool destroyed_out_of_order_{};  // only relevant on the root Storage
  InternalSharedPtr<Shape> shape_{};
  std::string_view provenance_{};  // only relevant on the root Storage

  std::int32_t level_{};
  std::optional<InternalSharedPtr<StoragePartition>> parent_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> color_{};
  SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets_{};

  std::size_t scalar_offset_{};

  // Storage is backed by one of three different types of data
  // noted in the variant below. However, the underlying types are optional because
  // Storage can take a two-stage initialization process where, first, the constructor
  // determines the type of backing for the Storage, and then, the user may set
  // the backing data later via the set_region_field, set_future, etc methods.
  // std::optional allows us to specify whether the specific variant alternative is set or not.
  std::variant<std::optional<InternalSharedPtr<LogicalRegionField>>,
               std::optional<Legion::Future>,
               std::optional<Legion::FutureMap>>
    storage_data_{std::optional<InternalSharedPtr<LogicalRegionField>>{}};

  std::uint32_t num_pieces_{};
  std::optional<InternalSharedPtr<Partition>> key_partition_{};
};

[[nodiscard]] InternalSharedPtr<Storage> slice_storage(
  const InternalSharedPtr<Storage>& self,
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> tile_shape,
  SmallVector<std::int64_t, LEGATE_MAX_DIM> offsets);

[[nodiscard]] InternalSharedPtr<StoragePartition> create_storage_partition(
  const InternalSharedPtr<Storage>& self,
  InternalSharedPtr<Partition> partition,
  std::optional<bool> complete);

}  // namespace legate::detail

#include <legate/data/detail/storage.inl>
