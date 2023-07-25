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

#pragma once

#include "library.h"
#include "tree_reduce_cffi.h"

namespace tree_reduce {

struct ReduceNormalTask : public Task<ReduceNormalTask, REDUCE_NORMAL> {
  static void cpu_variant(legate::TaskContext& context);
};

}  // namespace tree_reduce
