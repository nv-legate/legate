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

#include "produce_unbound.h"

namespace tree_reduce {

/*static*/ void ProduceUnboundTask::cpu_variant(legate::TaskContext& context)
{
  auto& output = context.outputs().at(0);
  auto size    = context.get_task_index()[0] + 1;
  auto buffer  = output.create_output_buffer<int64_t, 1>(legate::Point<1>(size), true /*bind*/);
  for (int64_t idx = 0; idx < size; ++idx) buffer[idx] = size;
}

}  // namespace tree_reduce
