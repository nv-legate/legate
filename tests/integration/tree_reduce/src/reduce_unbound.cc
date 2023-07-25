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

#include "reduce_unbound.h"

namespace tree_reduce {

/*static*/ void ReduceUnboundTask::cpu_variant(legate::TaskContext& context)
{
  auto& inputs      = context.inputs();
  auto& output      = context.outputs().at(0);
  uint32_t expected = 1;
  for (auto& input : inputs) {
    auto shape = input.shape<1>();
    assert(shape.volume() == expected);
    ++expected;
  }
  output.create_output_buffer<int64_t, 1>(legate::Point<1>(0), true);
}

}  // namespace tree_reduce
