/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/restriction.h>

#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/zip.h>

#include <algorithm>
#include <stdexcept>

namespace legate::detail {

Restriction join(Restriction lhs, Restriction rhs) { return std::max(lhs, rhs); }

Restrictions join(const Restrictions& lhs, const Restrictions& rhs)
{
  auto result = lhs;

  join_inplace(result, rhs);
  return result;
}

void join_inplace(Restrictions& lhs, const Restrictions& rhs)
{
  if (lhs.size() != rhs.size()) {
    throw TracedException<std::invalid_argument>{"Restrictions must have the same size"};
  }
  if (rhs.empty()) {
    return;
  }
  if (lhs.empty()) {
    lhs = rhs;
    return;
  }
  for (auto&& [lhsv, rhsv] : detail::zip_equal(lhs, rhs)) {
    lhsv = join(lhsv, rhsv);
  }
}

}  // namespace legate::detail
