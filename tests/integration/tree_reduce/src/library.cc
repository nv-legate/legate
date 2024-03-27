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

#include "produce_normal.h"
#include "produce_unbound.h"
#include "reduce_normal.h"
#include "reduce_unbound.h"

namespace tree_reduce {

static const char* const library_name = "tree_reduce";

Legion::Logger log_tree_reduce(library_name);

void registration_callback()
{
  auto context = legate::Runtime::get_runtime()->create_library(library_name);

  ProduceUnboundTask::register_variants(context);
  ReduceUnboundTask::register_variants(context);
  ProduceNormalTask::register_variants(context);
  ReduceNormalTask::register_variants(context);
}

}  // namespace tree_reduce

extern "C" {

void perform_registration(void) { tree_reduce::registration_callback(); }
}
