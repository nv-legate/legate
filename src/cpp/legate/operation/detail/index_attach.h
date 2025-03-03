/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <vector>

namespace legate::detail {

class IndexAttach final : public Operation {
 public:
  IndexAttach(std::uint64_t unique_id,
              InternalSharedPtr<LogicalRegionField> region_field,
              std::uint32_t dim,
              std::vector<Legion::LogicalRegion> subregions,
              std::vector<InternalSharedPtr<ExternalAllocation>> allocations,
              InternalSharedPtr<mapping::detail::DimOrdering> ordering);

  void launch() override;

  [[nodiscard]] Kind kind() const override;

 private:
  InternalSharedPtr<LogicalRegionField> region_field_{};
  std::uint32_t dim_{};
  std::vector<Legion::LogicalRegion> subregions_{};
  std::vector<InternalSharedPtr<ExternalAllocation>> allocations_{};
  InternalSharedPtr<mapping::detail::DimOrdering> ordering_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/index_attach.inl>
