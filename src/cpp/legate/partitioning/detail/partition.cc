/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partition.h>

#include <iostream>

namespace legate::detail {

std::ostream& operator<<(std::ostream& out, const Partition& partition)
{
  out << partition.to_string();
  return out;
}

}  // namespace legate::detail
