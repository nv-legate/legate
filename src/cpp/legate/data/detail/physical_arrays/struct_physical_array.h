/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_array.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <optional>

namespace legate::detail {

class PhysicalStore;
class Type;

class StructPhysicalArray final : public PhysicalArray {
 public:
  StructPhysicalArray(InternalSharedPtr<Type> type,
                      std::optional<InternalSharedPtr<PhysicalStore>> null_mask,
                      SmallVector<InternalSharedPtr<PhysicalArray>>&& fields);

  [[nodiscard]] std::int32_t dim() const override;
  [[nodiscard]] const InternalSharedPtr<Type>& type() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] bool valid() const override;

  [[nodiscard]] const InternalSharedPtr<PhysicalStore>& null_mask() const override;
  [[nodiscard]] InternalSharedPtr<PhysicalArray> child(std::uint32_t index) const override;
  void populate_stores(SmallVector<InternalSharedPtr<PhysicalStore>>& result) const override;

  [[nodiscard]] Domain domain() const override;
  void check_shape_dimension(std::int32_t dim) const override;

 private:
  InternalSharedPtr<Type> type_{};
  std::optional<InternalSharedPtr<PhysicalStore>> null_mask_{};
  SmallVector<InternalSharedPtr<PhysicalArray>> fields_{};
};

}  // namespace legate::detail

#include <legate/data/detail/physical_arrays/struct_physical_array.inl>
