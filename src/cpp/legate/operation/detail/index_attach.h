/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
