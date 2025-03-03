/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/external_allocation.h>
#include <legate/data/detail/logical_region_field.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/operation/detail/operation.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <cstdint>

namespace legate::detail {

class Attach final : public Operation {
 public:
  Attach(std::uint64_t unique_id,
         InternalSharedPtr<LogicalRegionField> region_field,
         std::uint32_t dim,
         InternalSharedPtr<ExternalAllocation> allocation,
         InternalSharedPtr<mapping::detail::DimOrdering> ordering);

  void launch() override;

  [[nodiscard]] Kind kind() const override;

 private:
  InternalSharedPtr<LogicalRegionField> region_field_{};
  std::uint32_t dim_{};
  InternalSharedPtr<ExternalAllocation> allocation_{};
  InternalSharedPtr<mapping::detail::DimOrdering> ordering_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/attach.inl>
