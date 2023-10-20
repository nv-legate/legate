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

#include "core/data/shape.h"
#include "core/utilities/memory.h"
#include "core/utilities/tuple.h"

/** @defgroup partitioning Partitioning
 */

/**
 * @file
 * @brief Class definitions for partitioning constraint language
 */
namespace legate {

class AutoTask;

namespace detail {
struct Constraint;
class Variable;
}  // namespace detail
extern template struct default_delete<detail::Constraint>;

/**
 * @ingroup partitioning
 * @brief Class for partition symbols
 */
class Variable {
 public:
  std::string to_string() const;

 public:
  Variable() = default;
  Variable(const detail::Variable* impl);
  ~Variable() = default;

 public:
  const detail::Variable* impl() const { return impl_; }

 private:
  const detail::Variable* impl_{nullptr};
};

/**
 * @ingroup partitioning
 * @brief A base class for partitioning constraints
 */
class Constraint {
 public:
  std::string to_string() const;

 public:
  Constraint() = default;
  Constraint(std::shared_ptr<detail::Constraint>&& impl);
  Constraint(const Constraint&)                = default;
  Constraint(Constraint&&) noexcept            = default;
  Constraint& operator=(const Constraint&)     = default;
  Constraint& operator=(Constraint&&) noexcept = default;
  ~Constraint()                                = default;

  const std::shared_ptr<detail::Constraint> impl() const { return impl_; }

 private:
  std::shared_ptr<detail::Constraint> impl_{nullptr};
};

/**
 * @ingroup partitioning
 * @brief Creates an alignment constraint on two variables
 *
 * @param lhs LHS variable
 * @param rhs RHS variable
 *
 * @return Alignment constraint
 */
Constraint align(Variable lhs, Variable rhs);

/**
 * @ingroup partitioning
 * @brief Creates a broadcast constraint on a variable.
 *
 * This constraint prevents all dimensions of the store from being partitioned.
 *
 * @param variable Partition symbol to constrain
 *
 * @return Broadcast constraint
 */
[[nodiscard]] Constraint broadcast(Variable variable);

/**
 * @ingroup partitioning
 * @brief Creates a broadcast constraint on a variable.
 *
 * @param variable Partition symbol to constrain
 * @param axes List of dimensions to broadcast
 *
 * @return Broadcast constraint
 *
 * @throw std::invalid_argument If the list of axes is empty
 */
Constraint broadcast(Variable variable, const tuple<int32_t>& axes);

/**
 * @ingroup partitioning
 * @brief Creates an image constraint between partitions.
 *
 * @param var_function Partition symbol for the function store
 * @param var_range Partition symbol of the store whose partition should be derived from the image
 *
 * @return Broadcast constraint
 */
Constraint image(Variable var_function, Variable var_range);

/**
 * @ingroup partitioning
 * @brief Creates a scaling constraint between partitions
 *
 * If two stores `A` and `B` are constrained by a scaling constraint
 *
 *   `legate::scale(S, pA, pB)`
 *
 * where `pA` and `pB ` are partition symbols for `A` and `B`, respectively, `A` and `B` will be
 * partitioned such that each pair of sub-stores `Ak` and `Bk` satisfy the following property:
 *
 * @f$\mathtt{S} \cdot \mathit{dom}(\mathtt{Ak}) \cap \mathit{dom}(\mathtt{B}) \subseteq @f$
 * @f$\mathit{dom}(\mathtt{Bk})@f$
 *
 * @param factors Scaling factors
 * @param var_smaller Partition symbol for the smaller store (i.e., the one whose extents are
 * scaled)
 * @param var_bigger Partition symbol for the bigger store
 *
 * @return Scaling constraint
 */
Constraint scale(const Shape& factors, Variable var_smaller, Variable var_bigger);

/**
 * @ingroup partitioning
 * @brief Creates a bloating constraint between partitions
 *
 * If two stores `A` and `B` are constrained by a bloating constraint
 *
 *   `legate::bloat(pA, pB, L, H)`
 *
 * where `pA` and `pB ` are partition symbols for `A` and `B`, respectively, `A` and `B` will be
 * partitioned such that each pair of sub-stores `Ak` and `Bk` satisfy the following property:
 *
 * @f$ \forall p \in \mathit{dom}(\mathtt{Ak}). \forall \delta \in [-\mathtt{L}, \mathtt{H}]. @f$
 * @f$ p + \delta \in \mathit{dom}(\mathtt{Bk}) \lor p + \delta \not \in \mathit{dom}(\mathtt{B})@f$
 *
 * @param var_source Partition symbol for the source store
 * @param var_bloat Partition symbol for the target store
 * @param low_offsets Offsets to bloat towards the negative direction
 * @param high_offsets Offsets to bloat towards the positive direction
 *
 * @return Bloating constraint
 */
Constraint bloat(Variable var_source,
                 Variable var_bloat,
                 const Shape& low_offsets,
                 const Shape& high_offsets);

}  // namespace legate
