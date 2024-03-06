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

#include "core/operation/detail/scatter_gather.h"

namespace legate::detail {

inline void ScatterGather::set_source_indirect_out_of_range(bool flag)
{
  source_indirect_out_of_range_ = flag;
}

inline void ScatterGather::set_target_indirect_out_of_range(bool flag)
{
  target_indirect_out_of_range_ = flag;
}

inline Operation::Kind ScatterGather::kind() const { return Kind::SCATTER_GATHER; }

}  // namespace legate::detail
