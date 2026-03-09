/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/region_field.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/data/inline_allocation.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <optional>
#include <utility>

namespace legate::mapping {

enum class StoreTarget : std::uint8_t;

}

namespace legate::detail {

class RegionPhysicalStore final : public PhysicalStore {
 public:
  RegionPhysicalStore(std::int32_t dim,
                      InternalSharedPtr<Type> type,
                      GlobalRedopID redop_id,
                      RegionField&& region_field,
                      InternalSharedPtr<detail::TransformStack> transform = nullptr);

  [[nodiscard]] bool valid() const override;
  [[nodiscard]] Domain domain() const override;
  [[nodiscard]] InlineAllocation get_inline_allocation() const override;
  [[nodiscard]] mapping::StoreTarget target() const override;
  [[nodiscard]] bool is_partitioned() const override;

  /**
   * @brief Returns the backing logical region.
   * @return The logical region.
   */
  [[nodiscard]] std::optional<Legion::LogicalRegion> get_logical_region() const override;

  /**
   * @brief Returns the backing field ID.
   * @return The field ID.
   */
  [[nodiscard]] std::optional<Legion::FieldID> get_field_id() const override;

  /**
   * @brief Creates a LogicalStore wrapping this region-backed store.
   * @param self Shared pointer to this store.
   * @return LogicalStore wrapping the same backing storage.
   */
  [[nodiscard]] InternalSharedPtr<LogicalStore> to_logical_store(
    const InternalSharedPtr<PhysicalStore>& self) const override;

  // region field specific interface
  [[nodiscard]] std::pair<Legion::PhysicalRegion, Legion::FieldID> get_region_field() const;

 private:
  RegionField region_field_{};
};

}  // namespace legate::detail

#include <legate/data/detail/physical_stores/region_physical_store.inl>
