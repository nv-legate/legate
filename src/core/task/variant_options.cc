/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/task/variant_options.h"

namespace legate {

VariantOptions& VariantOptions::with_leaf(bool _leaf)
{
  leaf = _leaf;
  return *this;
}

VariantOptions& VariantOptions::with_inner(bool _inner)
{
  inner = _inner;
  return *this;
}

VariantOptions& VariantOptions::with_idempotent(bool _idempotent)
{
  idempotent = _idempotent;
  return *this;
}

VariantOptions& VariantOptions::with_concurrent(bool _concurrent)
{
  concurrent = _concurrent;
  return *this;
}

VariantOptions& VariantOptions::with_return_size(std::size_t _return_size)
{
  return_size = _return_size;
  return *this;
}

void VariantOptions::populate_registrar(Legion::TaskVariantRegistrar& registrar) const
{
  registrar.set_leaf(leaf);
  registrar.set_inner(inner);
  registrar.set_idempotent(idempotent);
  registrar.set_concurrent(concurrent);
}

std::ostream& operator<<(std::ostream& os, const VariantOptions& options)
{
  std::stringstream ss;
  ss << "(";
  if (options.leaf) {
    ss << "leaf,";
  }
  if (options.concurrent) {
    ss << "concurrent,";
  }
  ss << options.return_size << ")";
  os << std::move(ss).str();
  return os;
}

}  // namespace legate
