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

#include "core/operation/projection.h"

#include "core/utilities/hash.h"

#include <fmt/format.h>

namespace legate {

std::string SymbolicExpr::to_string() const
{
  std::string result;

  if (weight_ != 0) {
    if (weight_ != 1) {
      fmt::format_to(std::back_inserter(result), "{}*", weight_);
    }
    fmt::format_to(std::back_inserter(result), "COORD{}", dim_);
  }
  if (offset_ != 0) {
    fmt::format_to(std::back_inserter(result), "{}{}", offset_ > 0 ? "+" : "-", offset_);
  }
  return result;
}

std::size_t SymbolicExpr::hash() const { return hash_all(dim_, weight_, offset_); }

std::ostream& operator<<(std::ostream& out, const SymbolicExpr& expr)
{
  out << expr.to_string();
  return out;
}

}  // namespace legate
