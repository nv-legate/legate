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

#include "legate/task/variant_options.h"

namespace legate {

void VariantOptions::populate_registrar(Legion::TaskVariantRegistrar& registrar) const
{
  registrar.set_leaf(true);
  registrar.set_inner(false);
  registrar.set_idempotent(false);
  registrar.set_concurrent(concurrent);
  registrar.set_concurrent_barrier(concurrent);
}

std::ostream& operator<<(std::ostream& os, const VariantOptions& options)
{
  os << "(";
  if (options.concurrent) {
    os << "concurrent,";
  }
  if (options.has_allocations) {
    os << "has_allocations,";
  }
  if (options.elide_device_ctx_sync) {
    os << "elide_device_ctx_sync";
  }
  os << ")";
  return os;
}

}  // namespace legate
