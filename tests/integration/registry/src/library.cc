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

#include "library.h"
#include "core/mapping/mapping.h"
#include "world.h"

namespace rg {

static const char* const library_name = "registry";

Legion::Logger log_registry(library_name);

/*static*/ legate::TaskRegistrar& Registry::get_registrar()
{
  static legate::TaskRegistrar registrar;
  return registrar;
}

void registration_callback()
{
  auto context = legate::Runtime::get_runtime()->create_library(library_name);

  // Task registration via a registrar
  Registry::get_registrar().register_all_tasks(context);
  // Immediate task registration
  WorldTask::register_variants(context);
}

}  // namespace rg

extern "C" {

void perform_registration(void) { rg::registration_callback(); }
}
