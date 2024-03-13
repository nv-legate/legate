/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/data/buffer.h"
#include "core/data/detail/transform.h"
#include "core/data/inline_allocation.h"
#include "core/mapping/mapping.h"
#include "core/task/detail/return.h"
#include "core/type/detail/type_info.h"
#include "core/utilities/internal_shared_ptr.h"

#include <memory>

namespace legate {
class PhysicalStore;
}  // namespace legate

namespace legate::detail {

class BasePhysicalArray;

class RegionField {
 public:
  RegionField() = default;
  RegionField(std::int32_t dim, const Legion::PhysicalRegion& pr, Legion::FieldID fid);

  RegionField(RegionField&& other) noexcept            = default;
  RegionField& operator=(RegionField&& other) noexcept = default;

  RegionField(const RegionField& other)            = delete;
  RegionField& operator=(const RegionField& other) = delete;

  [[nodiscard]] bool valid() const;

  [[nodiscard]] std::int32_t dim() const;

  [[nodiscard]] Domain domain() const;
  void set_logical_region(const Legion::LogicalRegion& lr);
  [[nodiscard]] InlineAllocation get_inline_allocation(std::uint32_t field_size) const;
  [[nodiscard]] InlineAllocation get_inline_allocation(
    std::uint32_t field_size,
    const Domain& domain,
    const Legion::DomainAffineTransform& transform) const;
  [[nodiscard]] mapping::StoreTarget target() const;

  [[nodiscard]] bool is_readable() const;
  [[nodiscard]] bool is_writable() const;
  [[nodiscard]] bool is_reducible() const;

  [[nodiscard]] Legion::PhysicalRegion get_physical_region() const;
  [[nodiscard]] Legion::FieldID get_field_id() const;

 private:
  std::int32_t dim_{-1};
  std::unique_ptr<Legion::PhysicalRegion> pr_{};
  Legion::LogicalRegion lr_{};
  Legion::FieldID fid_{-1U};

  bool readable_{};
  bool writable_{};
  bool reducible_{};
};

class UnboundRegionField {
 public:
  UnboundRegionField() = default;
  UnboundRegionField(const Legion::OutputRegion& out, Legion::FieldID fid);

  UnboundRegionField(UnboundRegionField&& other) noexcept;
  UnboundRegionField& operator=(UnboundRegionField&& other) noexcept;

  UnboundRegionField(const UnboundRegionField& other)            = delete;
  UnboundRegionField& operator=(const UnboundRegionField& other) = delete;

  [[nodiscard]] bool bound() const;

  void bind_empty_data(std::int32_t dim);

  [[nodiscard]] ReturnValue pack_weight() const;

  void set_bound(bool bound);
  void update_num_elements(std::size_t num_elements);

  [[nodiscard]] Legion::OutputRegion get_output_region() const;
  [[nodiscard]] Legion::FieldID get_field_id() const;

 private:
  bool bound_{};
  Legion::UntypedDeferredValue num_elements_{};
  Legion::OutputRegion out_{};
  Legion::FieldID fid_{-1U};
};

class FutureWrapper {
 public:
  FutureWrapper() = default;
  FutureWrapper(bool read_only,
                std::uint32_t field_size,
                const Domain& domain,
                Legion::Future future,
                bool initialize = false);

  [[nodiscard]] std::int32_t dim() const;
  [[nodiscard]] Domain domain() const;
  [[nodiscard]] bool valid() const;

  [[nodiscard]] InlineAllocation get_inline_allocation(const Domain& domain) const;
  [[nodiscard]] InlineAllocation get_inline_allocation() const;
  [[nodiscard]] mapping::StoreTarget target() const;

  void initialize_with_identity(std::int32_t redop_id);

  [[nodiscard]] ReturnValue pack() const;

  [[nodiscard]] bool is_read_only() const;
  [[nodiscard]] Legion::Future get_future() const;
  [[nodiscard]] Legion::UntypedDeferredValue get_buffer() const;

 private:
  bool read_only_{true};
  std::uint32_t field_size_{};
  Domain domain_{};
  std::unique_ptr<Legion::Future> future_{};
  Legion::UntypedDeferredValue buffer_{};
};

class PhysicalStore {
 public:
  PhysicalStore(std::int32_t dim,
                InternalSharedPtr<Type> type,
                std::int32_t redop_id,
                FutureWrapper future,
                InternalSharedPtr<detail::TransformStack>&& transform = nullptr);
  PhysicalStore(std::int32_t dim,
                InternalSharedPtr<Type> type,
                std::int32_t redop_id,
                RegionField&& region_field,
                InternalSharedPtr<detail::TransformStack>&& transform = nullptr);
  PhysicalStore(std::int32_t dim,
                InternalSharedPtr<Type> type,
                UnboundRegionField&& unbound_field,
                InternalSharedPtr<detail::TransformStack>&& transform = nullptr);
  PhysicalStore(std::int32_t dim,
                InternalSharedPtr<Type> type,
                std::int32_t redop_id,
                FutureWrapper future,
                const InternalSharedPtr<detail::TransformStack>& transform);
  PhysicalStore(std::int32_t dim,
                InternalSharedPtr<Type> type,
                std::int32_t redop_id,
                RegionField&& region_field,
                const InternalSharedPtr<detail::TransformStack>& transform);

  PhysicalStore(PhysicalStore&& other) noexcept            = default;
  PhysicalStore& operator=(PhysicalStore&& other) noexcept = default;

  PhysicalStore(const PhysicalStore& other)            = delete;
  PhysicalStore& operator=(const PhysicalStore& other) = delete;

  [[nodiscard]] bool valid() const;
  [[nodiscard]] bool transformed() const;

  [[nodiscard]] std::int32_t dim() const;
  [[nodiscard]] const InternalSharedPtr<Type>& type() const;

  [[nodiscard]] Domain domain() const;
  [[nodiscard]] InlineAllocation get_inline_allocation() const;
  [[nodiscard]] mapping::StoreTarget target() const;

  [[nodiscard]] bool is_readable() const;
  [[nodiscard]] bool is_writable() const;
  [[nodiscard]] bool is_reducible() const;

  void bind_empty_data();

  [[nodiscard]] bool is_future() const;
  [[nodiscard]] bool is_unbound_store() const;
  [[nodiscard]] ReturnValue pack() const;
  [[nodiscard]] ReturnValue pack_weight() const;

 private:
  friend class legate::PhysicalStore;
  friend class legate::detail::BasePhysicalArray;
  void check_accessor_dimension(std::int32_t dim) const;
  void check_buffer_dimension(std::int32_t dim) const;
  void check_shape_dimension(std::int32_t dim) const;
  void check_valid_binding(bool bind_buffer) const;
  void check_write_access() const;
  void check_reduction_access() const;

  [[nodiscard]] Legion::DomainAffineTransform get_inverse_transform() const;

  void get_region_field(Legion::PhysicalRegion& pr, Legion::FieldID& fid) const;
  [[nodiscard]] std::int32_t get_redop_id() const;

  [[nodiscard]] bool is_read_only_future() const;
  [[nodiscard]] Legion::Future get_future() const;
  [[nodiscard]] Legion::UntypedDeferredValue get_buffer() const;

  void get_output_field(Legion::OutputRegion& out, Legion::FieldID& fid);
  void update_num_elements(std::size_t num_elements);

  bool is_future_{};
  bool is_unbound_store_{};
  std::int32_t dim_{-1};
  InternalSharedPtr<Type> type_{};
  std::int32_t redop_id_{-1};

  FutureWrapper future_{};
  RegionField region_field_{};
  UnboundRegionField unbound_field_{};

  InternalSharedPtr<detail::TransformStack> transform_{};

  bool readable_{};
  bool writable_{};
  bool reducible_{};
};

}  // namespace legate::detail

#include "core/data/detail/physical_store.inl"
