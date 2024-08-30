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

#include "core/operation/detail/unmap_and_detach.h"

#include "core/runtime/detail/runtime.h"

namespace legate::detail {

void UnmapAndDetach::launch()
{
  physical_state_->unmap_and_detach(unordered_);
  physical_state_->invoke_callbacks();
}

}  // namespace legate::detail
