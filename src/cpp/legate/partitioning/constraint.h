/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/partitioning/proxy.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/shared_ptr.h>
#include <legate/utilities/tuple.h>

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

/**
 * @file
 * @brief Class definitions for partitioning constraint language
 */

namespace legate {

/**
 * @addtogroup partitioning
 * @{
 */

namespace detail {

class Constraint;
class Variable;
class ProxyConstraint;

}  // namespace detail

/**
 * @brief Class for partition symbols
 */
class Variable {
 public:
  Variable() = LEGATE_DEFAULT_WHEN_CYTHON;

  explicit Variable(const detail::Variable* impl);

  [[nodiscard]] std::string to_string() const;

  [[nodiscard]] const detail::Variable* impl() const;

 private:
  const detail::Variable* impl_{};
};

/**
 * @brief A base class for partitioning constraints
 */
class Constraint {
 public:
  Constraint() = LEGATE_DEFAULT_WHEN_CYTHON;

  explicit Constraint(InternalSharedPtr<detail::Constraint>&& impl);

  [[nodiscard]] std::string to_string() const;

  [[nodiscard]] const SharedPtr<detail::Constraint>& impl() const;

 private:
  SharedPtr<detail::Constraint> impl_{};
};

/**
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
 * @brief Construct an alignment constraint descriptor from a pair of proxy objects.
 *
 * This routine may be used to describe an alignment constraint between prospective arguments
 * to a task. For example:
 *
 * @snippet{trimleft} unit/task_signature/register.cc Align all inputs with output 0
 *
 * Dictates that all inputs should be aligned with output `0`. Similarly
 *
 * @snippet{trimleft} unit/task_signature/register.cc Align all input 0 with output 1
 *
 * Dictates that inputs `0` and `1` of the task should be aligned.
 *
 * @param left The left operand to the alignment constraint.
 * @param right The right operand to the alignment constraint.
 *
 * @return The alignment descriptor.
 *
 * @see align(Variable, Variable)
 */
[[nodiscard]] ProxyConstraint align(
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      left,
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      right);

/**
 * @brief Construct an alignment constraint descriptor for all input arguments.
 *
 * The returned constraint aligns all input arguments with each other.
 *
 * @param proxies The input arguments.
 *
 * @return The alignment descriptor.
 */
[[nodiscard]] ProxyConstraint align(ProxyInputArguments proxies);

/**
 * @brief Construct an alignment constraint descriptor for all output arguments.
 *
 * The returned constraint aligns all output arguments with each other.
 *
 * @param proxies The output arguments.
 *
 * @return The alignment descriptor.
 */
[[nodiscard]] ProxyConstraint align(ProxyOutputArguments proxies);

/**
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
 * @brief Construct a broadcast constraint descriptor.
 *
 * This routine may be used to describe a broadcast constraint for prospective arguments to a
 * task. For example:
 *
 * @snippet{trimleft} unit/task_signature/register.cc Broadcast input 0
 *
 * Dictates that the first input argument should be broadcast to all leaf tasks, while
 *
 * @snippet{trimleft} unit/task_signature/register.cc Broadcast all outputs
 *
 * Dictates that all outputs should be broadcast to all leaf tasks.
 *
 * See `legate::broadcast()` for more information on the precise semantics of broadcasting
 * arguments.
 *
 * @param value The proxy value to apply the broadcast constraint to.
 * @param axes Optional axes to specify when broadcasting.
 *
 * @return The broadcast descriptor.
 *
 * @see broadcast(Variable, tuple<std::uint32_t>)
 */
[[nodiscard]] ProxyConstraint broadcast(std::variant<ProxyArrayArgument,
                                                     ProxyInputArguments,
                                                     ProxyOutputArguments,
                                                     ProxyReductionArguments> value,
                                        std::optional<tuple<std::uint32_t>> axes = std::nullopt);

/**
 * @brief Hints to the runtime for the image computation
 */
enum class ImageComputationHint : std::uint8_t {
  NO_HINT,    /*!< A precise image of the function is needed */
  MIN_MAX,    /*!< An approximate image of the function using bounding boxes is sufficient */
  FIRST_LAST, /*!< Elements in the function store are sorted and thus bounding can be computed
                     using only the first and the last elements */
};

/**
 * @brief Creates an image constraint between partitions.
 *
 * The elements of \p var_function are treated as pointers to elements in \p var_range. Each
 * sub-store `s` of \p var_function is aligned with a sub-store `t` of \p var_range, such that
 * every element in `s` will find the element of \p var_range it's pointing to inside of `t`.
 *
 * @param var_function Partition symbol for the function store
 * @param var_range Partition symbol of the store whose partition should be derived from the image
 * @param hint Optional hint to the runtime describing how the image computation can be performed.
 * If no hint is given (which is the default), the runtime falls back to the precise image
 * computation. Otherwise, the runtime computes a potentially approximate image of the function.
 *
 * @return Image constraint
 *
 * @note An approximate image of a function potentially contains extra points not in the function's
 * image. For example, if a function sub-store contains two 2-D points (0, 0) and (1, 1), the
 * corresponding sub-store of the range would only contain the elements at points (0, 0) and (1, 1)
 * if it was constructed from a precise image computation, whereas an approximate image computation
 * would yield a sub-store with elements at point (0, 0), (0, 1), (1, 0), and (1, 1) (two extra
 * elements).
 *
 * Currently, the precise image computation can be performed only by CPUs. As a result, the
 * function store is copied to the system memory if the store was last updated by GPU tasks.
 * The approximate image computation has no such issue and is fully GPU accelerated.
 *
 */
[[nodiscard]] Constraint image(Variable var_function,
                               Variable var_range,
                               ImageComputationHint hint = ImageComputationHint::NO_HINT);

/**
 * @brief Construct an image constraint descriptor.
 *
 * This routine may be used to describe an image constraint for prospective arguments to a
 * task.
 *
 * See `legate::image()` for more information on the precise semantics of image constraints.
 *
 * @param var_function The proxy symbol for the function store.
 * @param var_range The proxy symbol for the range store.
 * @param hint The optional hint given to the runtime describing how the image computation
 * will be performed.
 *
 * @return The image descriptor.
 *
 * @see image(Variable, Variable, ImageComputationHint)
 */
[[nodiscard]] ProxyConstraint image(
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      var_function,
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      var_range,
  std::optional<ImageComputationHint> hint = std::nullopt);

/**
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
 * @brief Construct a scaling constraint descriptor.
 *
 * This routine may be used to describe a scaling constraint for prospective arguments to a
 * task.
 *
 * See `legate::scale()` for more information on the precise semantics of scaling constraints.
 *
 * @param factors The scaling factors.
 * @param var_smaller The proxy argument for the smaller store (that which should be scaled).
 * @param var_bigger The proxy argument for the bigger store.
 *
 * @return The scale descriptor.
 *
 * @see scale(tuple<std::uint64_t>, Variable, Variable)
 */
[[nodiscard]] ProxyConstraint scale(
  tuple<std::uint64_t> factors,
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      var_smaller,
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      var_bigger);

/**
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

/**
 * @brief Construct a bloat constraint descriptor.
 *
 * This routine may be used to describe a bloat constraint for prospective arguments to a
 * task.
 *
 * See `legate::bloat()` for more information on the precise semantics of bloat constraints.
 *
 * @param var_source The proxy source store.
 * @param var_bloat The proxy target store.
 * @param low_offsets Offsets to bloat towards the negative direction.
 * @param high_offsets Offsets to bloat towards the positive direction.
 *
 * @return The bloat descriptor.
 *
 * @see bloat(Variable, Variable, tuple<std::uint64_t>, tuple<std::uint64_t>)
 */
[[nodiscard]] ProxyConstraint bloat(
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      var_source,
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      var_bloat,
  tuple<std::uint64_t> low_offsets,
  tuple<std::uint64_t> high_offsets);

/** @} */

}  // namespace legate

#include <legate/partitioning/constraint.inl>
