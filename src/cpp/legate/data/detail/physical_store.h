/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/inline_storage.h>
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

class PhysicalStore {
 public:
  PhysicalStore(std::int32_t dim,
                InternalSharedPtr<Type> type,
                InternalSharedPtr<detail::TransformStack> transform,
                Legion::PrivilegeMode priv,
                InternalSharedPtr<InlineStorage> storage,
                const Domain& domain);

  PhysicalStore(PhysicalStore&& other) noexcept            = default;
  PhysicalStore& operator=(PhysicalStore&& other) noexcept = default;

  PhysicalStore(const PhysicalStore& other)            = delete;
  PhysicalStore& operator=(const PhysicalStore& other) = delete;

  ~PhysicalStore() = default;

  [[nodiscard]] bool valid() const;
  [[nodiscard]] bool transformed() const;

  [[nodiscard]] std::int32_t dim() const;
  [[nodiscard]] const InternalSharedPtr<Type>& type() const;

  /**
   * @brief Returns the store's domain as a Rect<DIM>
   *
   * @return Store's domain as a Rect<DIM>
   */
  [[nodiscard]] Domain domain() const;
  [[nodiscard]] InlineAllocation get_inline_allocation() const;
  [[nodiscard]] mapping::StoreTarget target() const;
  [[nodiscard]] bool is_accessible(mapping::TaskTarget task_target) const;

  [[nodiscard]] bool is_readable() const;
  [[nodiscard]] bool is_writable() const;
  [[nodiscard]] bool is_reducible() const;

  [[nodiscard]] Legion::DomainAffineTransform get_inverse_transform() const;

  void check_shape_dimension(std::int32_t dim) const;
  void check_accessor_type(Type::Code code, std::size_t size_of_T) const;
  void check_accessor_dimension(std::int32_t dim) const;
  void check_write_access() const;
  void check_reduction_access() const;

  /**
   * @brief Returns the region instance and field ID wrapping the underlying memory allocation.
   *
   * This is useful for information such as constructing Legion accessors for interaction
   * with the underlying memory.
   *
   * @return std::pair holding the region instance and field ID wrapping wrapped memory allocation.
   */
  [[nodiscard]] std::pair<Realm::RegionInstance, Realm::FieldID> get_region_instance() const;

 protected:
  InternalSharedPtr<detail::TransformStack> transform_{};
  InternalSharedPtr<Type> type_{};

 private:
  [[nodiscard]] const InternalSharedPtr<InlineStorage>& storage_() const;

  std::int32_t dim_{-1};

  bool readable_{};
  bool writable_{};
  bool reducible_{};

  // The inline storage that holds the manually-managed memory of the store.
  InternalSharedPtr<InlineStorage> inline_storage_{};

  // The domain over inline_storage_ that this store spans.
  Domain domain_{};
};

}  // namespace legate::detail

#include <legate/data/detail/physical_store.inl>
