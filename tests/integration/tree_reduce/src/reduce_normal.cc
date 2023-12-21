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

#include "reduce_normal.h"

namespace tree_reduce {

/*static*/ void ReduceNormalTask::cpu_variant(legate::TaskContext context)
{
  auto inputs = context.inputs();
  auto output = context.output(0);
  for (auto& input : inputs) {
    auto shape = input.shape<1>();
    assert(shape.empty() || shape.volume() == TILE_SIZE);
  }
  output.data().create_output_buffer<int64_t, 1>(legate::Point<1>(0), true);
}

}  // namespace tree_reduce
