/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/release_region_field.h>

#include <legate/runtime/detail/runtime.h>

namespace legate::detail {

void ReleaseRegionField::launch()
{
  physical_state_->unmap();
  physical_state_->detach(unordered_);
  physical_state_->invoke_callbacks();
}

bool ReleaseRegionField::supports_streaming() const
{
  return !(physical_state_->physical_region().exists() || physical_state_->has_callbacks());
}

}  // namespace legate::detail
