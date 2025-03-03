/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/discard.h>

#include <legate/runtime/detail/runtime.h>

namespace legate::detail {

void Discard::launch()
{
  auto* runtime = Runtime::get_runtime();
  auto launcher = Legion::DiscardLauncher{region_, region_};

  launcher.add_field(field_id_);
  static_assert(std::is_same_v<decltype(launcher.provenance), std::string>,
                "Don't use to_string() below");
  launcher.provenance = provenance().to_string();
  runtime->get_legion_runtime()->discard_fields(runtime->get_legion_context(), launcher);
}

}  // namespace legate::detail
