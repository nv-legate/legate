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

#include "core/partitioning/detail/partitioner.h"

namespace legate::detail {

inline bool Strategy::parallel(const Operation* op) const { return launch_domain(op).is_valid(); }

// ==========================================================================================

inline Partitioner::Partitioner(std::vector<Operation*>&& operations)
  : operations_{std::move(operations)}
{
}

}  // namespace legate::detail
