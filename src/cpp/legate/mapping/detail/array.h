/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/array_kind.h>
#include <legate/mapping/detail/store.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <optional>
#include <vector>

namespace legate::mapping::detail {

class Array {
 public:
  virtual ~Array() = default;

  [[nodiscard]] virtual std::int32_t dim() const                                    = 0;
  [[nodiscard]] virtual legate::detail::ArrayKind kind() const                      = 0;
  [[nodiscard]] virtual const InternalSharedPtr<legate::detail::Type>& type() const = 0;
  [[nodiscard]] virtual bool unbound() const                                        = 0;
  [[nodiscard]] virtual bool nullable() const                                       = 0;
  [[nodiscard]] virtual bool nested() const                                         = 0;
  [[nodiscard]] virtual bool valid() const                                          = 0;

  [[nodiscard]] virtual const InternalSharedPtr<Store>& data() const;
  [[nodiscard]] virtual const InternalSharedPtr<Store>& null_mask() const         = 0;
  [[nodiscard]] virtual InternalSharedPtr<Array> child(std::uint32_t index) const = 0;

  virtual void populate_stores(std::vector<InternalSharedPtr<Store>>& result) const = 0;
  [[nodiscard]] virtual Domain domain() const                                       = 0;
};

class BaseArray final : public Array {
 public:
  BaseArray(InternalSharedPtr<Store> data, std::optional<InternalSharedPtr<Store>> null_mask);

  [[nodiscard]] std::int32_t dim() const override;
  [[nodiscard]] legate::detail::ArrayKind kind() const override;
  [[nodiscard]] const InternalSharedPtr<legate::detail::Type>& type() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] bool valid() const override;

  [[nodiscard]] const InternalSharedPtr<Store>& data() const override;
  [[nodiscard]] const InternalSharedPtr<Store>& null_mask() const override;
  [[nodiscard]] InternalSharedPtr<Array> child(std::uint32_t index) const override;
  void populate_stores(std::vector<InternalSharedPtr<Store>>& result) const override;

  [[nodiscard]] Domain domain() const override;

 private:
  InternalSharedPtr<Store> data_{};
  std::optional<InternalSharedPtr<Store>> null_mask_{};
};

class ListArray final : public Array {
 public:
  ListArray(InternalSharedPtr<legate::detail::Type> type,
            InternalSharedPtr<BaseArray> descriptor,
            InternalSharedPtr<Array> vardata);

  [[nodiscard]] std::int32_t dim() const override;
  [[nodiscard]] legate::detail::ArrayKind kind() const override;
  [[nodiscard]] const InternalSharedPtr<legate::detail::Type>& type() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] bool valid() const override;

  [[nodiscard]] const InternalSharedPtr<Store>& null_mask() const override;
  [[nodiscard]] InternalSharedPtr<Array> child(std::uint32_t index) const override;
  void populate_stores(std::vector<InternalSharedPtr<Store>>& result) const override;
  [[nodiscard]] const InternalSharedPtr<BaseArray>& descriptor() const;
  [[nodiscard]] const InternalSharedPtr<Array>& vardata() const;
  [[nodiscard]] Domain domain() const override;

 private:
  InternalSharedPtr<legate::detail::Type> type_{};
  InternalSharedPtr<BaseArray> descriptor_{};
  InternalSharedPtr<Array> vardata_{};
};

class StructArray final : public Array {
 public:
  StructArray(InternalSharedPtr<legate::detail::Type> type,
              std::optional<InternalSharedPtr<Store>> null_mask,
              std::vector<InternalSharedPtr<Array>>&& fields);

  [[nodiscard]] std::int32_t dim() const override;
  [[nodiscard]] legate::detail::ArrayKind kind() const override;
  [[nodiscard]] const InternalSharedPtr<legate::detail::Type>& type() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] bool valid() const override;

  [[nodiscard]] const InternalSharedPtr<Store>& null_mask() const override;
  [[nodiscard]] InternalSharedPtr<Array> child(std::uint32_t index) const override;
  void populate_stores(std::vector<InternalSharedPtr<Store>>& result) const override;

  [[nodiscard]] Domain domain() const override;

  [[nodiscard]] const std::vector<InternalSharedPtr<Array>>& fields() const;

 private:
  InternalSharedPtr<legate::detail::Type> type_{};
  std::optional<InternalSharedPtr<Store>> null_mask_{};
  std::vector<InternalSharedPtr<Array>> fields_{};
};

}  // namespace legate::mapping::detail

#include <legate/mapping/detail/array.inl>
