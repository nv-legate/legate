/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/buffer.h>
#include <legate/data/detail/buffer.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/physical_stores/unbound_region_field.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/data/inline_allocation.h>
#include <legate/task/detail/return_value.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace legate::mapping {

enum class StoreTarget : std::uint8_t;

}

namespace legate::detail {

class UnboundPhysicalStore final : public PhysicalStore {
 public:
  UnboundPhysicalStore(std::int32_t dim,
                       InternalSharedPtr<Type> type,
                       UnboundRegionField&& unbound_field,
                       InternalSharedPtr<detail::TransformStack> transform = nullptr);

  [[nodiscard]] bool valid() const override;
  [[nodiscard]] Domain domain() const override;
  [[nodiscard]] InlineAllocation get_inline_allocation() const override;
  [[nodiscard]] mapping::StoreTarget target() const override;
  [[nodiscard]] bool is_partitioned() const override;

  // Unbound specific API
  void bind_empty_data();

  void bind_untyped_data(Buffer<std::int8_t, 1>& buffer, const Point<1>& extents);

  /**
   * @brief Binds a `TaskLocalBuffer` to the store.
   *
   * Valid only when the store is unbound and has not yet been bound to another
   * `TaskLocalBuffer`. The `TaskLocalBuffer` must be consistent with the mapping policy for
   * the store.  Recommend that the `TaskLocalBuffer` be created by a `create_output_buffer()`
   * call.
   *
   * Passing `extents` that are smaller than the actual extents of the `TaskLocalBuffer` is
   * legal; the runtime uses the passed extents as the extents of this store.
   *
   * @param buffer `TaskLocalBuffer` to bind to the store.
   * @param extents Extents of the `TaskLocalBuffer`.
   */
  void bind_data(const InternalSharedPtr<TaskLocalBuffer>& buffer, const DomainPoint& extents);

  [[nodiscard]] ReturnValue pack_weight() const;

  void check_valid_binding(bool bind_buffer) const;
  void check_buffer_dimension(std::int32_t dim) const;

  [[nodiscard]] std::pair<Legion::OutputRegion, Legion::FieldID> get_output_field() const;
  void update_num_elements(std::size_t num_elements);

 private:
  UnboundRegionField unbound_field_{};
};

}  // namespace legate::detail

#include <legate/data/detail/physical_stores/unbound_physical_store.inl>
