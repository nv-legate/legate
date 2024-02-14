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
#include "core/data/detail/physical_store.h"
#include "core/data/physical_array.h"
#include "core/utilities/internal_shared_ptr.h"

#include <vector>

namespace legate::detail {

struct PhysicalArray {
  virtual ~PhysicalArray()                                   = default;
  [[nodiscard]] virtual std::int32_t dim() const             = 0;
  [[nodiscard]] virtual ArrayKind kind() const               = 0;
  [[nodiscard]] virtual InternalSharedPtr<Type> type() const = 0;
  [[nodiscard]] virtual bool unbound() const                 = 0;
  [[nodiscard]] virtual bool nullable() const                = 0;
  [[nodiscard]] virtual bool nested() const                  = 0;
  [[nodiscard]] virtual bool valid() const                   = 0;

  [[nodiscard]] virtual InternalSharedPtr<PhysicalStore> data() const;
  [[nodiscard]] virtual InternalSharedPtr<PhysicalStore> null_mask() const                = 0;
  [[nodiscard]] virtual InternalSharedPtr<PhysicalArray> child(std::uint32_t index) const = 0;
  virtual void _stores(std::vector<InternalSharedPtr<PhysicalStore>>& result) const       = 0;

  [[nodiscard]] std::vector<InternalSharedPtr<PhysicalStore>> stores() const;

  [[nodiscard]] virtual Domain domain() const                = 0;
  virtual void check_shape_dimension(std::int32_t dim) const = 0;
};

class BasePhysicalArray final : public PhysicalArray {
 public:
  BasePhysicalArray(InternalSharedPtr<PhysicalStore> data,
                    InternalSharedPtr<PhysicalStore> null_mask);

  [[nodiscard]] std::int32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
  [[nodiscard]] InternalSharedPtr<Type> type() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] bool valid() const override;

  [[nodiscard]] InternalSharedPtr<PhysicalStore> data() const override;
  [[nodiscard]] InternalSharedPtr<PhysicalStore> null_mask() const override;
  [[nodiscard]] InternalSharedPtr<PhysicalArray> child(std::uint32_t index) const override;
  void _stores(std::vector<InternalSharedPtr<PhysicalStore>>& result) const override;

  [[nodiscard]] Domain domain() const override;
  void check_shape_dimension(std::int32_t dim) const override;

 private:
  InternalSharedPtr<PhysicalStore> data_{};
  InternalSharedPtr<PhysicalStore> null_mask_{};
};

class ListPhysicalArray final : public PhysicalArray {
 public:
  ListPhysicalArray(InternalSharedPtr<Type> type,
                    InternalSharedPtr<BasePhysicalArray> descriptor,
                    InternalSharedPtr<PhysicalArray> vardata);

  [[nodiscard]] std::int32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
  [[nodiscard]] InternalSharedPtr<Type> type() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] bool valid() const override;

  [[nodiscard]] InternalSharedPtr<PhysicalStore> null_mask() const override;
  [[nodiscard]] InternalSharedPtr<PhysicalArray> child(std::uint32_t index) const override;
  void _stores(std::vector<InternalSharedPtr<PhysicalStore>>& result) const override;
  [[nodiscard]] InternalSharedPtr<PhysicalArray> descriptor() const;
  [[nodiscard]] InternalSharedPtr<PhysicalArray> vardata() const;

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
                      InternalSharedPtr<PhysicalStore> null_mask,
                      std::vector<InternalSharedPtr<PhysicalArray>>&& fields);

  [[nodiscard]] std::int32_t dim() const override;
  [[nodiscard]] ArrayKind kind() const override;
  [[nodiscard]] InternalSharedPtr<Type> type() const override;
  [[nodiscard]] bool unbound() const override;
  [[nodiscard]] bool nullable() const override;
  [[nodiscard]] bool nested() const override;
  [[nodiscard]] bool valid() const override;

  [[nodiscard]] InternalSharedPtr<PhysicalStore> null_mask() const override;
  [[nodiscard]] InternalSharedPtr<PhysicalArray> child(std::uint32_t index) const override;
  void _stores(std::vector<InternalSharedPtr<PhysicalStore>>& result) const override;

  [[nodiscard]] Domain domain() const override;
  void check_shape_dimension(std::int32_t dim) const override;

 private:
  InternalSharedPtr<Type> type_{};
  InternalSharedPtr<PhysicalStore> null_mask_{};
  std::vector<InternalSharedPtr<PhysicalArray>> fields_{};
};

}  // namespace legate::detail

#include "core/data/detail/physical_array.inl"
