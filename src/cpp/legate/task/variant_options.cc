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
  registrar.set_leaf(leaf);
  registrar.set_inner(inner);
  registrar.set_idempotent(idempotent);
  registrar.set_concurrent(concurrent);
  if (concurrent) {
    registrar.set_concurrent_barrier(true);
  }
}

std::ostream& operator<<(std::ostream& os, const VariantOptions& options)
{
  os << "(";
  if (options.leaf) {
    os << "leaf,";
  }
  if (options.inner) {
    os << "inner,";
  }
  if (options.idempotent) {
    os << "idempotent,";
  }
  if (options.concurrent) {
    os << "concurrent,";
  }
  os << options.return_size << ")";
  return os;
}

}  // namespace legate
