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

#pragma once

#include <legate/task/variant_options.h>

namespace legate {

constexpr VariantOptions& VariantOptions::with_concurrent(bool _concurrent)
{
  concurrent = _concurrent;
  return *this;
}

constexpr VariantOptions& VariantOptions::with_has_allocations(bool _has_allocations)
{
  has_allocations = _has_allocations;
  return *this;
}

constexpr VariantOptions& VariantOptions::with_elide_device_ctx_sync(bool elide_sync)
{
  elide_device_ctx_sync = elide_sync;
  return *this;
}

constexpr bool VariantOptions::operator==(const VariantOptions& other) const
{
  return concurrent == other.concurrent && has_allocations == other.has_allocations &&
         elide_device_ctx_sync == other.elide_device_ctx_sync;
}

constexpr bool VariantOptions::operator!=(const VariantOptions& other) const
{
  return !(*this == other);
}

}  // namespace legate
