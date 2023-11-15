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

#include "core/partitioning/restriction.h"

#include <algorithm>

namespace legate {

Restriction join(Restriction lhs, Restriction rhs) { return std::max(lhs, rhs); }

tuple<Restriction> join(const tuple<Restriction>& lhs, const tuple<Restriction>& rhs)
{
  auto result = lhs;

  join_inplace(result, rhs);
  return result;
}

void join_inplace(Restrictions& lhs, const Restrictions& rhs)
{
  if (rhs.empty()) {
    return;
  }
  if (lhs.empty()) {
    lhs = rhs;
    return;
  }
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("Restrictions must have the same size");
  }
  for (uint32_t idx = 0; idx < lhs.size(); ++idx) {
    lhs[idx] = join(lhs[idx], rhs[idx]);
  }
}

}  // namespace legate
