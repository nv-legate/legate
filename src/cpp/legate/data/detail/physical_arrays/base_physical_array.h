/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_array.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <optional>

namespace legate::detail {

class Type;
class PhysicalStore;

template <typename T, std::uint32_t S>
class SmallVector;

class BasePhysicalArray final : public PhysicalArray {
 public:
  BasePhysicalArray(InternalSharedPtr<PhysicalStore> data,
                    std::optional<InternalSharedPtr<PhysicalStore>> null_mask);

  [[nodiscard]] std::int32_t dim() const override;
  [[nodiscard]] const InternalSharedPtr<Type>& type() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] bool valid() const override;

  [[nodiscard]] const InternalSharedPtr<PhysicalStore>& data() const override;
  [[nodiscard]] const InternalSharedPtr<PhysicalStore>& null_mask() const override;
  [[nodiscard]] InternalSharedPtr<PhysicalArray> child(std::uint32_t index) const override;
  void populate_stores(SmallVector<InternalSharedPtr<PhysicalStore>>& result) const override;

  [[nodiscard]] Domain domain() const override;
  void check_shape_dimension(std::int32_t dim) const override;

 private:
  InternalSharedPtr<PhysicalStore> data_{};
  std::optional<InternalSharedPtr<PhysicalStore>> null_mask_{};
};

}  // namespace legate::detail

#include <legate/data/detail/physical_arrays/base_physical_array.inl>
