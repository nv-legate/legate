/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/discard.h>

#include <legion.h>

namespace legate::detail {

void Discard::launch()
{
  auto launcher = Legion::DiscardLauncher{region(), region()};

  launcher.add_field(field_id());
  static_assert(std::is_same_v<decltype(launcher.provenance), std::string>,
                "Don't use to_string() below");
  launcher.provenance = provenance().to_string();
  Legion::Runtime::get_runtime()->discard_fields(Legion::Runtime::get_context(), launcher);
}

}  // namespace legate::detail
