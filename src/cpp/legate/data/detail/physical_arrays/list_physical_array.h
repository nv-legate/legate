/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/physical_array.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>

namespace legate::detail {

class PhysicalStore;
class BasePhysicalArray;
class Type;

template <typename T, std::uint32_t S>
class SmallVector;

class ListPhysicalArray final : public PhysicalArray {
 public:
  ListPhysicalArray(InternalSharedPtr<Type> type,
                    InternalSharedPtr<BasePhysicalArray> descriptor,
                    InternalSharedPtr<PhysicalArray> vardata);

  [[nodiscard]] std::int32_t dim() const override;
  [[nodiscard]] const InternalSharedPtr<Type>& type() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] bool valid() const override;

  [[nodiscard]] const InternalSharedPtr<PhysicalStore>& null_mask() const override;
  [[nodiscard]] InternalSharedPtr<PhysicalArray> child(std::uint32_t index) const override;
  void populate_stores(SmallVector<InternalSharedPtr<PhysicalStore>>& result) const override;
  [[nodiscard]] const InternalSharedPtr<BasePhysicalArray>& descriptor() const;
  [[nodiscard]] const InternalSharedPtr<PhysicalArray>& vardata() const;

  [[nodiscard]] Domain domain() const override;
  void check_shape_dimension(std::int32_t dim) const override;

 private:
  InternalSharedPtr<Type> type_{};
  InternalSharedPtr<BasePhysicalArray> descriptor_{};
  InternalSharedPtr<PhysicalArray> vardata_{};
};

}  // namespace legate::detail

#include <legate/data/detail/physical_arrays/list_physical_array.inl>
