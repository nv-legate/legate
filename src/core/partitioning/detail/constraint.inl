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

#pragma once

#include "core/partitioning/detail/constraint.h"
#include "core/utilities/hash.h"

namespace legate::detail {

inline Variable::Variable(const Operation* op, std::int32_t id) : op_{op}, id_{id} {}

inline bool operator==(const Variable& lhs, const Variable& rhs)
{
  return lhs.op_ == rhs.op_ && lhs.id_ == rhs.id_;
}

inline bool Variable::closed() const { return false; }

inline const Operation* Variable::operation() const { return op_; }

inline std::int32_t Variable::id() const { return id_; }

inline std::size_t Variable::hash() const noexcept { return static_cast<std::size_t>(id()); }

// ==========================================================================================

inline Alignment::Alignment(const Variable* lhs, const Variable* rhs) : lhs_{lhs}, rhs_{rhs} {}

inline Alignment::Kind Alignment::kind() const { return Kind::ALIGNMENT; }

inline const Variable* Alignment::lhs() const { return lhs_; }

inline const Variable* Alignment::rhs() const { return rhs_; }

inline bool Alignment::is_trivial() const { return *lhs_ == *rhs_; }

// ==========================================================================================

inline Broadcast::Broadcast(const Variable* variable, tuple<std::uint32_t> axes)
  : variable_{variable}, axes_{std::move(axes)}
{
}

inline Broadcast::Broadcast(const Variable* variable) : Broadcast{variable, tuple<std::uint32_t>{}}
{
}

inline Broadcast::Kind Broadcast::kind() const { return Kind::BROADCAST; }

inline const Variable* Broadcast::variable() const { return variable_; }

inline const tuple<std::uint32_t>& Broadcast::axes() const { return axes_; }

// ==========================================================================================

inline ImageConstraint::ImageConstraint(const Variable* var_function,
                                        const Variable* var_range,
                                        ImageComputationHint hint)
  : var_function_{var_function}, var_range_{var_range}, hint_{hint}
{
}

inline ImageConstraint::Kind ImageConstraint::kind() const { return Kind::IMAGE; }

inline const Variable* ImageConstraint::var_function() const { return var_function_; }

inline const Variable* ImageConstraint::var_range() const { return var_range_; }

// ==========================================================================================

inline ScaleConstraint::ScaleConstraint(tuple<std::uint64_t> factors,
                                        const Variable* var_smaller,
                                        const Variable* var_bigger)
  : factors_{std::move(factors)}, var_smaller_{var_smaller}, var_bigger_{var_bigger}
{
}

inline ScaleConstraint::Kind ScaleConstraint::kind() const { return Kind::SCALE; }

inline const tuple<std::uint64_t>& ScaleConstraint::factors() const { return factors_; }

inline const Variable* ScaleConstraint::var_smaller() const { return var_smaller_; }

inline const Variable* ScaleConstraint::var_bigger() const { return var_bigger_; }

// ==========================================================================================

inline BloatConstraint::BloatConstraint(const Variable* var_source,
                                        const Variable* var_bloat,
                                        tuple<std::uint64_t> low_offsets,
                                        tuple<std::uint64_t> high_offsets)
  : var_source_{var_source},
    var_bloat_{var_bloat},
    low_offsets_{std::move(low_offsets)},
    high_offsets_{std::move(high_offsets)}
{
}

inline BloatConstraint::Kind BloatConstraint::kind() const { return Kind::BLOAT; }

inline const Variable* BloatConstraint::var_source() const { return var_source_; }

inline const Variable* BloatConstraint::var_bloat() const { return var_bloat_; }

inline const tuple<std::uint64_t>& BloatConstraint::low_offsets() const { return low_offsets_; }

inline const tuple<std::uint64_t>& BloatConstraint::high_offsets() const { return high_offsets_; }

}  // namespace legate::detail

namespace std {

template <>
struct hash<legate::detail::Variable> {
  [[nodiscard]] std::size_t operator()(const legate::detail::Variable& v) const noexcept
  {
    return v.hash();
  }
};

}  // namespace std
