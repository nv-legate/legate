/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/variant_options.h>

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
    os << "elide_device_ctx_sync,";
  }
  if (options.has_side_effect) {
    os << "has_side_effect,";
  }
  if (options.may_throw_exception) {
    os << "may_throw_exceptions,";
  }
  if (const auto& comms = options.communicators; comms.has_value()) {
    os << "communicator(";
    for (auto&& c : *comms) {
      if (c.empty()) {
        break;
      }
      os << c << ",";
    }
    os << ")";
  }
  os << ")";
  return os;
}

}  // namespace legate
