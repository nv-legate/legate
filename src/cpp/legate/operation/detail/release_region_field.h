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

#include "legate/data/detail/logical_region_field.h"
#include "legate/operation/detail/operation.h"
#include "legate/utilities/internal_shared_ptr.h"

#include <cstdint>

namespace legate::detail {

class ReleaseRegionField final : public Operation {
 public:
  ReleaseRegionField(std::uint64_t unique_id,
                     InternalSharedPtr<LogicalRegionField::PhysicalState> physical_state,
                     bool unmap,
                     bool unordered);

  void launch() override;

  [[nodiscard]] Kind kind() const override;

 private:
  InternalSharedPtr<LogicalRegionField::PhysicalState> physical_state_{};
  bool unmap_{};
  bool unordered_{};
};

}  // namespace legate::detail

#include "legate/operation/detail/release_region_field.inl"
