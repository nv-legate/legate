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

#include "core/data/array.h"
#include "core/data/detail/array_kind.h"
#include "core/data/detail/store.h"

#include <memory>
#include <vector>

namespace legate::detail {

struct Array {
  virtual ~Array()                                         = default;
  [[nodiscard]] virtual int32_t dim() const                = 0;
  [[nodiscard]] virtual ArrayKind kind() const             = 0;
  [[nodiscard]] virtual std::shared_ptr<Type> type() const = 0;
  [[nodiscard]] virtual bool unbound() const               = 0;
  [[nodiscard]] virtual bool nullable() const              = 0;
  [[nodiscard]] virtual bool nested() const                = 0;
  [[nodiscard]] virtual bool valid() const                 = 0;

  [[nodiscard]] virtual std::shared_ptr<Store> data() const;
  [[nodiscard]] virtual std::shared_ptr<Store> null_mask() const           = 0;
  [[nodiscard]] virtual std::shared_ptr<Array> child(uint32_t index) const = 0;
  virtual void _stores(std::vector<std::shared_ptr<Store>>& result) const  = 0;

  [[nodiscard]] std::vector<std::shared_ptr<Store>> stores() const;

  [[nodiscard]] virtual Domain domain() const           = 0;
  virtual void check_shape_dimension(int32_t dim) const = 0;
};

class BaseArray : public Array {
 public:
  BaseArray(std::shared_ptr<Store> data, std::shared_ptr<Store> null_mask);

  [[nodiscard]] int32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
  [[nodiscard]] std::shared_ptr<Type> type() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] bool valid() const override;

  [[nodiscard]] std::shared_ptr<Store> data() const override;
  [[nodiscard]] std::shared_ptr<Store> null_mask() const override;
  [[nodiscard]] std::shared_ptr<Array> child(uint32_t index) const override;
  void _stores(std::vector<std::shared_ptr<Store>>& result) const override;

  [[nodiscard]] Domain domain() const override;
  void check_shape_dimension(int32_t dim) const override;

 private:
  std::shared_ptr<Store> data_{};
  std::shared_ptr<Store> null_mask_{};
};

class ListArray : public Array {
 public:
  ListArray(std::shared_ptr<Type> type,
            std::shared_ptr<BaseArray> descriptor,
            std::shared_ptr<Array> vardata);

  [[nodiscard]] int32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
  [[nodiscard]] std::shared_ptr<Type> type() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] bool valid() const override;

  [[nodiscard]] std::shared_ptr<Store> null_mask() const override;
  [[nodiscard]] std::shared_ptr<Array> child(uint32_t index) const override;
  void _stores(std::vector<std::shared_ptr<Store>>& result) const override;
  [[nodiscard]] std::shared_ptr<Array> descriptor() const;
  [[nodiscard]] std::shared_ptr<Array> vardata() const;

  [[nodiscard]] Domain domain() const override;
  void check_shape_dimension(int32_t dim) const override;

 private:
  std::shared_ptr<Type> type_{};
  std::shared_ptr<BaseArray> descriptor_{};
  std::shared_ptr<Array> vardata_{};
};

class StructArray : public Array {
 public:
  StructArray(std::shared_ptr<Type> type,
              std::shared_ptr<Store> null_mask,
              std::vector<std::shared_ptr<Array>>&& fields);

  [[nodiscard]] int32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
  [[nodiscard]] std::shared_ptr<Type> type() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] bool valid() const override;

  [[nodiscard]] std::shared_ptr<Store> null_mask() const override;
  [[nodiscard]] std::shared_ptr<Array> child(uint32_t index) const override;
  void _stores(std::vector<std::shared_ptr<Store>>& result) const override;

  [[nodiscard]] Domain domain() const override;
  void check_shape_dimension(int32_t dim) const override;

 private:
  std::shared_ptr<Type> type_{};
  std::shared_ptr<Store> null_mask_{};
  std::vector<std::shared_ptr<Array>> fields_{};
};

}  // namespace legate::detail

#include "core/data/detail/array.inl"
