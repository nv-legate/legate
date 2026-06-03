/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partition.h>

#include <iostream>

namespace legate::detail {

Domain Partition::shard(const Domain&, const DomainPoint&) const
{
  LEGATE_ABORT("Sharding not implemented for this partition type");
  return {};
}

std::ostream& operator<<(std::ostream& out, const Partition& partition)
{
  out << partition.to_string();
  return out;
}

}  // namespace legate::detail
