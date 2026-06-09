/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/partition/no_partition.h>

namespace legate::detail {

inline bool NoPartition::is_convertible() const { return true; }

inline bool NoPartition::is_invertible() const { return true; }

inline bool NoPartition::has_color_shape() const { return false; }

inline Domain NoPartition::shard(const Domain& domain, const DomainPoint&) const { return domain; }

}  // namespace legate::detail
