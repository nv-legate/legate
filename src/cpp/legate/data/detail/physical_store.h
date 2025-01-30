/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/data/detail/future_wrapper.h>
#include <legate/data/detail/region_field.h>
#include <legate/data/detail/transform.h>
#include <legate/data/inline_allocation.h>
#include <legate/mapping/mapping.h>
#include <legate/task/detail/return_value.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <cstdint>

namespace legate {
class PhysicalStore;
}  // namespace legate

namespace legate::detail {

class BasePhysicalArray;

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

class PhysicalStore {
 public:
  PhysicalStore(std::int32_t dim,
                InternalSharedPtr<Type> type,
                GlobalRedopID redop_id,
                FutureWrapper future,
                InternalSharedPtr<detail::TransformStack> transform = nullptr);
  PhysicalStore(std::int32_t dim,
                InternalSharedPtr<Type> type,
                GlobalRedopID redop_id,
                RegionField&& region_field,
                InternalSharedPtr<detail::TransformStack> transform = nullptr);
  PhysicalStore(std::int32_t dim,
                InternalSharedPtr<Type> type,
                UnboundRegionField&& unbound_field,
                InternalSharedPtr<detail::TransformStack> transform = nullptr);

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
  [[nodiscard]] const Legion::Future& get_future() const;
  [[nodiscard]] const Legion::UntypedDeferredValue& get_buffer() const;

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
  void check_accessor_dimension_(std::int32_t dim) const;
  void check_buffer_dimension_(std::int32_t dim) const;
  void check_shape_dimension_(std::int32_t dim) const;
  void check_valid_binding_(bool bind_buffer) const;
  void check_write_access_() const;
  void check_reduction_access_() const;

  [[nodiscard]] Legion::DomainAffineTransform get_inverse_transform_() const;

  void get_region_field_(Legion::PhysicalRegion& pr, Legion::FieldID& fid) const;
  [[nodiscard]] GlobalRedopID get_redop_id_() const;

  [[nodiscard]] bool is_read_only_future_() const;
  [[nodiscard]] std::size_t get_field_offset_() const;
  [[nodiscard]] const void* get_untyped_pointer_from_future_() const;

  void get_output_field_(Legion::OutputRegion& out, Legion::FieldID& fid);
  void update_num_elements_(std::size_t num_elements);

  bool is_future_{};
  bool is_unbound_store_{};
  std::int32_t dim_{-1};
  InternalSharedPtr<Type> type_{};
  GlobalRedopID redop_id_{-1};

  FutureWrapper future_{};
  RegionField region_field_{};
  UnboundRegionField unbound_field_{};

  InternalSharedPtr<detail::TransformStack> transform_{};

  bool readable_{};
  bool writable_{};
  bool reducible_{};
};

}  // namespace legate::detail

#include <legate/data/detail/physical_store.inl>
