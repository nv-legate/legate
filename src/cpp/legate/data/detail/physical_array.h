/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>

namespace legate::detail {

class Type;
class PhysicalStore;

class PhysicalArray {
 public:
  virtual ~PhysicalArray()                                          = default;
  [[nodiscard]] virtual std::int32_t dim() const                    = 0;
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

}  // namespace legate::detail
