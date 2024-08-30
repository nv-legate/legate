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

#include "core/data/detail/external_allocation.h"
#include "core/data/detail/logical_region_field.h"
#include "core/mapping/detail/mapping.h"
#include "core/operation/detail/operation.h"
#include "core/utilities/internal_shared_ptr.h"

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

#include "core/operation/detail/attach.inl"
