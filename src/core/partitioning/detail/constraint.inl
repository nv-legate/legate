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

#include "core/partitioning/detail/constraint.h"
#include "core/utilities/hash.h"

namespace legate::detail {

inline bool Literal::closed() const { return true; }

inline Literal::Kind Literal::kind() const { return Kind::LITERAL; }

inline const Literal* Literal::as_literal() const { return this; }

inline const Variable* Literal::as_variable() const { return nullptr; }

inline const InternalSharedPtr<Partition>& Literal::partition() const { return partition_; }

// ==========================================================================================

inline Variable::Variable(const Operation* op, int32_t id) : op_{op}, id_{id} {}

inline bool operator==(const Variable& lhs, const Variable& rhs)
{
  return lhs.op_ == rhs.op_ && lhs.id_ == rhs.id_;
}

inline bool Variable::closed() const { return false; }

inline Variable::Kind Variable::kind() const { return Kind::VARIABLE; }

inline const Literal* Variable::as_literal() const { return nullptr; }

inline const Variable* Variable::as_variable() const { return this; }

inline const Operation* Variable::operation() const { return op_; }

inline size_t Variable::hash() const noexcept { return hash_all(id_); }

// ==========================================================================================

inline Alignment::Alignment(const Variable* lhs, const Variable* rhs) : lhs_{lhs}, rhs_{rhs} {}

inline Alignment::Kind Alignment::kind() const { return Kind::ALIGNMENT; }

inline const Alignment* Alignment::as_alignment() const { return this; }

inline const Broadcast* Alignment::as_broadcast() const { return nullptr; }

inline const ImageConstraint* Alignment::as_image_constraint() const { return nullptr; }

inline const ScaleConstraint* Alignment::as_scale_constraint() const { return nullptr; }

inline const BloatConstraint* Alignment::as_bloat_constraint() const { return nullptr; }

inline const Variable* Alignment::lhs() const { return lhs_; }

inline const Variable* Alignment::rhs() const { return rhs_; }

inline bool Alignment::is_trivial() const { return *lhs_ == *rhs_; }

// ==========================================================================================

inline Broadcast::Broadcast(const Variable* variable, tuple<uint32_t> axes)
  : variable_{variable}, axes_{std::move(axes)}
{
}

inline Broadcast::Broadcast(const Variable* variable) : Broadcast{variable, tuple<uint32_t>{}} {}

inline Broadcast::Kind Broadcast::kind() const { return Kind::BROADCAST; }

inline const Alignment* Broadcast::as_alignment() const { return nullptr; }

inline const Broadcast* Broadcast::as_broadcast() const { return this; }

inline const ImageConstraint* Broadcast::as_image_constraint() const { return nullptr; }

inline const ScaleConstraint* Broadcast::as_scale_constraint() const { return nullptr; }

inline const BloatConstraint* Broadcast::as_bloat_constraint() const { return nullptr; }

inline const Variable* Broadcast::variable() const { return variable_; }

inline const tuple<uint32_t>& Broadcast::axes() const { return axes_; }

// ==========================================================================================

inline ImageConstraint::ImageConstraint(const Variable* var_function, const Variable* var_range)
  : var_function_{var_function}, var_range_{var_range}
{
}

inline ImageConstraint::Kind ImageConstraint::kind() const { return Kind::IMAGE; }

inline const Alignment* ImageConstraint::as_alignment() const { return nullptr; }

inline const Broadcast* ImageConstraint::as_broadcast() const { return nullptr; }

inline const ImageConstraint* ImageConstraint::as_image_constraint() const { return this; }

inline const ScaleConstraint* ImageConstraint::as_scale_constraint() const { return nullptr; }

inline const BloatConstraint* ImageConstraint::as_bloat_constraint() const { return nullptr; }

inline const Variable* ImageConstraint::var_function() const { return var_function_; }

inline const Variable* ImageConstraint::var_range() const { return var_range_; }

// ==========================================================================================

inline ScaleConstraint::ScaleConstraint(tuple<uint64_t> factors,
                                        const Variable* var_smaller,
                                        const Variable* var_bigger)
  : factors_{std::move(factors)}, var_smaller_{var_smaller}, var_bigger_{var_bigger}
{
}

inline ScaleConstraint::Kind ScaleConstraint::kind() const { return Kind::SCALE; }

inline const Alignment* ScaleConstraint::as_alignment() const { return nullptr; }

inline const Broadcast* ScaleConstraint::as_broadcast() const { return nullptr; }

inline const ImageConstraint* ScaleConstraint::as_image_constraint() const { return nullptr; }

inline const ScaleConstraint* ScaleConstraint::as_scale_constraint() const { return this; }

inline const BloatConstraint* ScaleConstraint::as_bloat_constraint() const { return nullptr; }

inline const Variable* ScaleConstraint::var_smaller() const { return var_smaller_; }

inline const Variable* ScaleConstraint::var_bigger() const { return var_bigger_; }

// ==========================================================================================

inline BloatConstraint::BloatConstraint(const Variable* var_source,
                                        const Variable* var_bloat,
                                        tuple<uint64_t> low_offsets,
                                        tuple<uint64_t> high_offsets)
  : var_source_{var_source},
    var_bloat_{var_bloat},
    low_offsets_{std::move(low_offsets)},
    high_offsets_{std::move(high_offsets)}
{
}

inline BloatConstraint::Kind BloatConstraint::kind() const { return Kind::BLOAT; }

inline const Alignment* BloatConstraint::as_alignment() const { return nullptr; }

inline const Broadcast* BloatConstraint::as_broadcast() const { return nullptr; }

inline const ImageConstraint* BloatConstraint::as_image_constraint() const { return nullptr; }

inline const ScaleConstraint* BloatConstraint::as_scale_constraint() const { return nullptr; }

inline const BloatConstraint* BloatConstraint::as_bloat_constraint() const { return this; }

inline const Variable* BloatConstraint::var_source() const { return var_source_; }

inline const Variable* BloatConstraint::var_bloat() const { return var_bloat_; }

}  // namespace legate::detail

namespace std {

template <>
struct hash<const legate::detail::Variable> {
  [[nodiscard]] size_t operator()(const legate::detail::Variable& v) const noexcept
  {
    return v.hash();
  }
};

}  // namespace std
