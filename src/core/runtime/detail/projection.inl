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

#include "core/runtime/detail/projection.h"

namespace legate::proj {

inline SymbolicExpr::SymbolicExpr(int32_t dim, int32_t weight, int32_t offset)
  : dim_{dim}, weight_{weight}, offset_{offset}
{
}

inline SymbolicExpr::SymbolicExpr(int32_t dim) : SymbolicExpr{dim, 1} {}

inline int32_t SymbolicExpr::dim() const { return dim_; }

inline int32_t SymbolicExpr::weight() const { return weight_; }

inline int32_t SymbolicExpr::offset() const { return offset_; }

inline bool SymbolicExpr::is_identity(int32_t dim) const
{
  return this->dim() == dim && weight() == 1 && offset() == 0;
}

inline bool SymbolicExpr::operator==(const SymbolicExpr& other) const
{
  return dim() == other.dim() && weight() == other.weight() && offset() == other.offset();
}

inline bool SymbolicExpr::operator<(const SymbolicExpr& other) const
{
  if (dim() < other.dim()) return true;
  if (dim() > other.dim()) return false;
  if (weight() < other.weight()) return true;
  if (weight() > other.weight()) return false;
  if (offset() < other.offset()) return true;
  return false;
}

inline SymbolicExpr SymbolicExpr::operator*(int32_t other) const
{
  return {dim(), weight() * other, offset() * other};
}

inline SymbolicExpr SymbolicExpr::operator+(int32_t other) const
{
  return {dim(), weight(), offset() + other};
}

// ==========================================================================================

inline RadixProjectionFunctor::RadixProjectionFunctor(int32_t radix, int32_t offset)
  : offset_{offset}, radix_{radix}
{
}

}  // namespace legate::proj

namespace legate::detail {

inline LegateProjectionFunctor::LegateProjectionFunctor(Legion::Runtime* runtime)
  : ProjectionFunctor{runtime}
{
}

inline bool LegateProjectionFunctor::is_functional() const { return true; }

inline bool LegateProjectionFunctor::is_exclusive() const { return true; }

inline unsigned LegateProjectionFunctor::get_depth() const { return 0; }

}  // namespace legate::detail
