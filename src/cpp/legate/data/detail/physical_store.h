/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/transform/transform_stack.h>
#include <legate/data/inline_allocation.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace legate::mapping {

enum class StoreTarget : std::uint8_t;

}

namespace legate::detail {

class LogicalStore;
class RegionPhysicalStore;
class UnboundPhysicalStore;
class FuturePhysicalStore;
class InlinePhysicalStore;

class PhysicalStore {
 public:
  PhysicalStore(std::int32_t dim,
                InternalSharedPtr<Type> type,
                GlobalRedopID redop_id,
                InternalSharedPtr<detail::TransformStack> transform,
                bool readable,
                bool writable,
                bool reducible);

  PhysicalStore(PhysicalStore&& other) noexcept            = default;
  PhysicalStore& operator=(PhysicalStore&& other) noexcept = default;

  PhysicalStore(const PhysicalStore& other)            = delete;
  PhysicalStore& operator=(const PhysicalStore& other) = delete;

  virtual ~PhysicalStore() = default;

  [[nodiscard]] virtual bool valid() const = 0;
  [[nodiscard]] bool transformed() const;

  [[nodiscard]] std::int32_t dim() const;
  [[nodiscard]] const InternalSharedPtr<Type>& type() const;

  /**
   * @brief Returns the store's domain as a Rect<DIM>
   *
   * @return Store's domain as a Rect<DIM>
   */
  [[nodiscard]] virtual Domain domain() const                          = 0;
  [[nodiscard]] virtual InlineAllocation get_inline_allocation() const = 0;
  [[nodiscard]] virtual mapping::StoreTarget target() const            = 0;

  [[nodiscard]] bool is_readable() const;
  [[nodiscard]] bool is_writable() const;
  [[nodiscard]] bool is_reducible() const;

  [[nodiscard]] virtual bool is_partitioned() const = 0;

  // Cast to underlying implementations as abstraction is leaky
  [[nodiscard]] RegionPhysicalStore& as_region_store();
  [[nodiscard]] const RegionPhysicalStore& as_region_store() const;
  [[nodiscard]] UnboundPhysicalStore& as_unbound_store();
  [[nodiscard]] const UnboundPhysicalStore& as_unbound_store() const;
  [[nodiscard]] FuturePhysicalStore& as_future_store();
  [[nodiscard]] const FuturePhysicalStore& as_future_store() const;
  /**
   * @brief Cast to the underlying implementation as an inline storage store.
   *
   * @throw std::bad_cast If the underlying implementation is not an inline storage store.
   * @return The underlying implementation as an inline storage store.
   */
  [[nodiscard]] InlinePhysicalStore& as_inline_store();
  /**
   * @brief Cast to the underlying implementation as a const inline storage store.
   *
   * @throw std::bad_cast If the underlying implementation is not an inline storage store.
   * @return The underlying implementation as a const inline storage store.
   */
  [[nodiscard]] const InlinePhysicalStore& as_inline_store() const;

  [[nodiscard]] Legion::DomainAffineTransform get_inverse_transform() const;
  [[nodiscard]] GlobalRedopID get_redop_id() const;

  void check_shape_dimension(std::int32_t dim) const;
  void check_accessor_type(Type::Code code, std::size_t size_of_T) const;
  void check_accessor_dimension(std::int32_t dim) const;
  void check_accessor_store_backing() const;
  void check_write_access() const;
  void check_reduction_access() const;
  void check_scalar_store() const;
  void check_unbound_store() const;

  /**
   * @brief Get the logical region if this is a region-backed store.
   * @return The logical region, or std::nullopt for non-region stores.
   */
  [[nodiscard]] virtual std::optional<Legion::LogicalRegion> get_logical_region() const;

  /**
   * @brief Get the field ID if this is a region-backed store.
   * @return The field ID, or std::nullopt for non-region stores.
   */
  [[nodiscard]] virtual std::optional<Legion::FieldID> get_field_id() const;

  /**
   * @brief Creates a LogicalStore wrapping this PhysicalStore.
   * @param self Shared pointer to this store.
   * @return LogicalStore wrapping the same backing storage.
   * @throws std::runtime_error if not supported for this store type.
   */
  [[nodiscard]] virtual InternalSharedPtr<LogicalStore> to_logical_store(
    const InternalSharedPtr<PhysicalStore>& self) const = 0;

 protected:
  InternalSharedPtr<detail::TransformStack> transform_{};
  InternalSharedPtr<Type> type_{};

 private:
  std::int32_t dim_{-1};
  GlobalRedopID redop_id_{-1};

  bool readable_{};
  bool writable_{};
  bool reducible_{};
};

}  // namespace legate::detail

#include <legate/data/detail/physical_store.inl>
