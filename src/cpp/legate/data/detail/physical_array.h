/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/array_kind.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/physical_array.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <cstdint>
#include <optional>

namespace legate::detail {

class PhysicalArray {
 public:
  virtual ~PhysicalArray()                                          = default;
  [[nodiscard]] virtual std::int32_t dim() const                    = 0;
  [[nodiscard]] virtual ArrayKind kind() const                      = 0;
  [[nodiscard]] virtual const InternalSharedPtr<Type>& type() const = 0;
  [[nodiscard]] virtual bool unbound() const                        = 0;
  [[nodiscard]] virtual bool nullable() const                       = 0;
  [[nodiscard]] virtual bool nested() const                         = 0;
  [[nodiscard]] virtual bool valid() const                          = 0;

  [[nodiscard]] virtual const InternalSharedPtr<PhysicalStore>& data() const;
  [[nodiscard]] virtual const InternalSharedPtr<PhysicalStore>& null_mask() const           = 0;
  [[nodiscard]] virtual InternalSharedPtr<PhysicalArray> child(std::uint32_t index) const   = 0;
  virtual void populate_stores(SmallVector<InternalSharedPtr<PhysicalStore>>& result) const = 0;

  [[nodiscard]] virtual Domain domain() const                = 0;
  virtual void check_shape_dimension(std::int32_t dim) const = 0;
};

class BasePhysicalArray final : public PhysicalArray {
 public:
  BasePhysicalArray(InternalSharedPtr<PhysicalStore> data,
                    std::optional<InternalSharedPtr<PhysicalStore>> null_mask);

  [[nodiscard]] std::int32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
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

class ListPhysicalArray final : public PhysicalArray {
 public:
  ListPhysicalArray(InternalSharedPtr<Type> type,
                    InternalSharedPtr<BasePhysicalArray> descriptor,
                    InternalSharedPtr<PhysicalArray> vardata);

  [[nodiscard]] std::int32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
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

class StructPhysicalArray final : public PhysicalArray {
 public:
  StructPhysicalArray(InternalSharedPtr<Type> type,
                      std::optional<InternalSharedPtr<PhysicalStore>> null_mask,
                      SmallVector<InternalSharedPtr<PhysicalArray>>&& fields);

  [[nodiscard]] std::int32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
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

#include <legate/data/detail/physical_array.inl>
