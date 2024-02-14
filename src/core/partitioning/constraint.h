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
#include "core/utilities/internal_shared_ptr.h"
#include "core/utilities/memory.h"
#include "core/utilities/shared_ptr.h"
#include "core/utilities/tuple.h"

#include <string>

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
  Variable() = default;
  explicit Variable(const detail::Variable* impl);

  [[nodiscard]] std::string to_string() const;

  [[nodiscard]] const detail::Variable* impl() const;

 private:
  const detail::Variable* impl_{};
};

/**
 * @ingroup partitioning
 * @brief A base class for partitioning constraints
 */
class Constraint {
 public:
  [[nodiscard]] std::string to_string() const;

  Constraint() = default;
  explicit Constraint(InternalSharedPtr<detail::Constraint>&& impl);
  Constraint(const Constraint&)                = default;
  Constraint(Constraint&&) noexcept            = default;
  Constraint& operator=(const Constraint&)     = default;
  Constraint& operator=(Constraint&&) noexcept = default;
  ~Constraint()                                = default;

  [[nodiscard]] const SharedPtr<detail::Constraint>& impl() const;

 private:
  SharedPtr<detail::Constraint> impl_{};
};

/**
 * @ingroup partitioning
 * @brief Creates an alignment constraint on two variables
 *
 * An alignment constraint between variables `x` and `y` indicates to the runtime that the
 * PhysicalStores (leaf-task-local portions, typically equal-size tiles) of the LogicalStores
 * corresponding to `x` and `y` must have the same global indices (i.e. the Stores must "align" with
 * one another).
 *
 * This is commonly used for e.g. element-wise operations. For example, consider an
 * element-wise addition (`z = x + y`), where each array is 100 elements long. Each leaf task
 * must receive the same local tile for all 3 arrays. For example, leaf task 0 receives indices
 * 0 - 24, leaf task 1 receives 25 - 49, leaf task 2 receives 50 - 74, and leaf task 3 receives
 * 75 - 99.
 *
 * @param lhs LHS variable
 * @param rhs RHS variable
 *
 * @return Alignment constraint
 */
[[nodiscard]] Constraint align(Variable lhs, Variable rhs);

/**
 * @ingroup partitioning
 * @brief Creates a broadcast constraint on a variable.
 *
 * A broadcast constraint informs the runtime that the variable should not be split among the
 * leaf tasks, instead, each leaf task should get a full copy of the underlying store. In other
 * words, the store should be "broadcast" in its entirety to all leaf tasks in a task launch.
 *
 * In effect, this constraint prevents all dimensions of the store from being partitioned.
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
 * A modified form of broadcast constraint which applies the broadcast to a subset of the axes of
 * the LogicalStore corresponding to \p variable. The Store will be partitioned on all other axes.
 *
 * @param variable Partition symbol to constrain
 * @param axes List of dimensions to broadcast
 *
 * @return Broadcast constraint
 *
 * @throw std::invalid_argument If the list of axes is empty
 */
[[nodiscard]] Constraint broadcast(Variable variable, tuple<std::uint32_t> axes);

/**
 * @ingroup partitioning
 * @brief Creates an image constraint between partitions.
 *
 * The elements of \p var_function are treated as pointers to elements in \p var_range. Each
 * sub-store `s` of \p var_function is aligned with a sub-store `t` of \p var_range, such that
 * every element in `s` will find the element of \p var_range it's pointing to inside of `t`.
 *
 * @param var_function Partition symbol for the function store
 * @param var_range Partition symbol of the store whose partition should be derived from the image
 *
 * @return Broadcast constraint
 */
[[nodiscard]] Constraint image(Variable var_function, Variable var_range);

/**
 * @ingroup partitioning
 * @brief Creates a scaling constraint between partitions
 *
 * A scaling constraint is similar to an alignment constraint, except that the sizes of the
 * aligned tiles is first scaled by \p factors.
 *
 * For example, this may be used in compacting a `5x56` array of `bool`s to a `5x7` array of bytes,
 * treated as a bitfield. In this case \p var_smaller would be the byte array, \p var_bigger would
 * be the array of `bool`s, and \p factors would be `[1, 8]` (a `2x3` tile on the byte array
 * corresponds to a `2x24` tile on the bool array.
 *
 * Formally: if two stores `A` and `B` are constrained by a scaling constraint
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
[[nodiscard]] Constraint scale(tuple<std::uint64_t> factors,
                               Variable var_smaller,
                               Variable var_bigger);

/**
 * @ingroup partitioning
 * @brief Creates a bloating constraint between partitions
 *
 * This is typically used in stencil computations, to instruct the runtime that the tiles on
 * the "private + ghost" partition (\p var_bloat) must align with the tiles on the "private"
 * partition (\p var_source), but also include a halo of additional elements off each end.
 *
 * For example, if \p var_source and \p var_bloat correspond to 10-element vectors, \p
 * var_source is split into 2 tiles, `0-4` and `5-9`, `low_offsets == 1` and `high_offsets ==
 * 2`, then \p var_bloat will be split into 2 tiles, `0-6` and `4-9`.
 *
 * Formally, if two stores `A` and `B` are constrained by a bloating constraint
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
[[nodiscard]] Constraint bloat(Variable var_source,
                               Variable var_bloat,
                               tuple<std::uint64_t> low_offsets,
                               tuple<std::uint64_t> high_offsets);

}  // namespace legate

#include "core/partitioning/constraint.inl"
