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
#include "collective_test.h"

namespace collective {

static const char* const library_name = "collective";

Legion::Logger log_collective(library_name);

void registration_callback()
{
  auto context = legate::Runtime::get_runtime()->create_library(library_name);

  CollectiveTestTask::register_variants(context);
}

}  // namespace collective

extern "C" {

void collective_perform_registration(void) { collective::registration_callback(); }
}
