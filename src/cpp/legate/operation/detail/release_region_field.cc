/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/operation/detail/release_region_field.h>

#include <legate/runtime/detail/runtime.h>

namespace legate::detail {

void ReleaseRegionField::launch()
{
  if (unmap_) {
    physical_state_->unmap_and_detach(unordered_);
  }
  physical_state_->invoke_callbacks();
}

}  // namespace legate::detail
