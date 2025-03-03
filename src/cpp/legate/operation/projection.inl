/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/projection.h>

namespace legate {

inline SymbolicExpr::SymbolicExpr(std::uint32_t dim, std::int32_t weight, std::int32_t offset)
  : dim_{dim}, weight_{weight}, offset_{offset}
{
}

inline SymbolicExpr::SymbolicExpr(std::uint32_t dim) : SymbolicExpr{dim, 1} {}

inline std::uint32_t SymbolicExpr::dim() const { return dim_; }

inline std::int32_t SymbolicExpr::weight() const { return weight_; }

inline std::int32_t SymbolicExpr::offset() const { return offset_; }

inline bool SymbolicExpr::is_identity(std::uint32_t dim) const
{
  return this->dim() == dim && weight() == 1 && offset() == 0;
}

inline bool SymbolicExpr::is_constant() const { return dim() == UNSET; }

inline bool SymbolicExpr::operator==(const SymbolicExpr& other) const
{
  return dim() == other.dim() && weight() == other.weight() && offset() == other.offset();
}

inline bool SymbolicExpr::operator<(const SymbolicExpr& other) const
{
  return std::make_tuple(dim(), weight(), offset()) <
         std::make_tuple(other.dim(), other.weight(), other.offset());
}

inline SymbolicExpr SymbolicExpr::operator*(std::int32_t other) const
{
  return {dim(), weight() * other, offset() * other};
}

inline SymbolicExpr SymbolicExpr::operator+(std::int32_t other) const
{
  return {dim(), weight(), offset() + other};
}

inline SymbolicExpr dimension(std::uint32_t dim) { return SymbolicExpr{dim}; }

inline SymbolicExpr constant(std::int32_t value) { return {SymbolicExpr::UNSET, 0, value}; }

}  // namespace legate
