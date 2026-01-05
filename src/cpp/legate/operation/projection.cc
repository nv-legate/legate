/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/projection.h>

#include <legate/utilities/hash.h>

#include <fmt/format.h>

namespace legate {

std::string SymbolicExpr::to_string() const
{
  std::string result;

  if (weight() != 0) {
    if (weight() != 1) {
      fmt::format_to(std::back_inserter(result), "{}*", weight());
    }
    fmt::format_to(std::back_inserter(result), "COORD{}", dim());
  }
  if (offset() != 0) {
    fmt::format_to(std::back_inserter(result), "{}{}", offset() > 0 ? "+" : "-", offset());
  }
  return result;
}

std::size_t SymbolicExpr::hash() const { return hash_all(dim(), weight(), offset()); }

std::ostream& operator<<(std::ostream& out, const SymbolicExpr& expr)
{
  out << expr.to_string();
  return out;
}

}  // namespace legate
