/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate/partitioning/detail/restriction.h"

#include "legate/utilities/detail/zip.h"
#include <legate/utilities/detail/traced_exception.h>

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
