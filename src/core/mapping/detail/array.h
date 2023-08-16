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

#include "core/data/detail/array_kind.h"
#include "core/mapping/detail/store.h"
#include "core/type/detail/type_info.h"

namespace legate::mapping::detail {

class Array {
 public:
  virtual ~Array() {}
  virtual int32_t dim() const                                = 0;
  virtual legate::detail::ArrayKind kind() const             = 0;
  virtual std::shared_ptr<legate::detail::Type> type() const = 0;
  virtual bool unbound() const                               = 0;
  virtual bool nullable() const                              = 0;
  virtual bool nested() const                                = 0;

 public:
  virtual std::shared_ptr<Store> data() const;
  virtual std::shared_ptr<Store> null_mask() const           = 0;
  virtual std::shared_ptr<Array> child(uint32_t index) const = 0;
  std::vector<std::shared_ptr<Store>> stores() const
  {
    std::vector<std::shared_ptr<Store>> result;
    _stores(result);
    return result;
  }
  virtual void _stores(std::vector<std::shared_ptr<Store>>& result) const = 0;

 public:
  virtual Domain domain() const = 0;

 protected:
  std::shared_ptr<Store> null_mask_{nullptr};
};

class BaseArray : public Array {
 public:
  BaseArray(std::shared_ptr<Store> data, std::shared_ptr<Store> null_mask);

 public:
  BaseArray(const BaseArray& other)            = default;
  BaseArray& operator=(const BaseArray& other) = default;
  BaseArray(BaseArray&& other)                 = default;
  BaseArray& operator=(BaseArray&& other)      = default;

 public:
  int32_t dim() const override { return data_->dim(); }
  legate::detail::ArrayKind kind() const override { return legate::detail::ArrayKind::BASE; }
  std::shared_ptr<legate::detail::Type> type() const override { return data_->type(); }
  bool unbound() const override;
  bool nullable() const override { return null_mask_ != nullptr; }
  bool nested() const override { return false; }

 public:
  std::shared_ptr<Store> data() const override { return data_; }
  std::shared_ptr<Store> null_mask() const override;
  std::shared_ptr<Array> child(uint32_t index) const override;
  void _stores(std::vector<std::shared_ptr<Store>>& result) const override;

 public:
  virtual Domain domain() const override;

 private:
  std::shared_ptr<Store> data_;
  std::shared_ptr<Store> null_mask_;
};

class ListArray : public Array {
 public:
  ListArray(std::shared_ptr<legate::detail::Type> type,
            std::shared_ptr<BaseArray> descriptor,
            std::shared_ptr<Array> vardata);

 public:
  ListArray(const ListArray& other)            = default;
  ListArray& operator=(const ListArray& other) = default;
  ListArray(ListArray&& other)                 = default;
  ListArray& operator=(ListArray&& other)      = default;

 public:
  int32_t dim() const override;
  legate::detail::ArrayKind kind() const override { return legate::detail::ArrayKind::LIST; }
  std::shared_ptr<legate::detail::Type> type() const override { return type_; }
  bool unbound() const override;
  bool nullable() const override { return vardata_->nullable(); }
  bool nested() const override { return true; }

 public:
  std::shared_ptr<Store> null_mask() const override { return descriptor_->null_mask(); }
  std::shared_ptr<Array> child(uint32_t index) const override;
  void _stores(std::vector<std::shared_ptr<Store>>& result) const override;
  std::shared_ptr<Array> descriptor() const { return descriptor_; }
  std::shared_ptr<Array> vardata() const { return vardata_; }

 public:
  virtual Domain domain() const override;

 private:
  std::shared_ptr<legate::detail::Type> type_;
  std::shared_ptr<BaseArray> descriptor_;
  std::shared_ptr<Array> vardata_;
};

class StructArray : public Array {
 public:
  StructArray(std::shared_ptr<legate::detail::Type> type,
              std::shared_ptr<Store> null_mask,
              std::vector<std::shared_ptr<Array>>&& fields);

 public:
  StructArray(const StructArray& other)            = default;
  StructArray& operator=(const StructArray& other) = default;
  StructArray(StructArray&& other)                 = default;
  StructArray& operator=(StructArray&& other)      = default;

 public:
  int32_t dim() const override;
  legate::detail::ArrayKind kind() const override { return legate::detail::ArrayKind::STRUCT; }
  std::shared_ptr<legate::detail::Type> type() const override { return type_; }
  bool unbound() const override;
  bool nullable() const override { return null_mask_ != nullptr; }
  bool nested() const override { return true; }

 public:
  std::shared_ptr<Store> null_mask() const override;
  std::shared_ptr<Array> child(uint32_t index) const override;
  void _stores(std::vector<std::shared_ptr<Store>>& result) const override;

 public:
  virtual Domain domain() const override;

 private:
  std::shared_ptr<legate::detail::Type> type_;
  std::shared_ptr<Store> null_mask_;
  std::vector<std::shared_ptr<Array>> fields_;
};

}  // namespace legate::mapping::detail
