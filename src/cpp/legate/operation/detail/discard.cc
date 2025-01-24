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
