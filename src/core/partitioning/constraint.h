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
  Variable(const detail::Variable* impl);
  ~Variable() = default;

 public:
  const detail::Variable* impl() const { return impl_; }

 private:
  const detail::Variable* impl_;
};

/**
 * @ingroup partitioning
 * @brief A base class for partitioning constraints
 */
class Constraint {
 public:
  std::string to_string() const;

 public:
  Constraint(detail::Constraint* impl);
  ~Constraint() = default;

 private:
  friend class AutoTask;
  detail::Constraint* release();

 private:
  std::unique_ptr<detail::Constraint, default_delete<detail::Constraint>> impl_;
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
 * @brief Creates a broadcast constraint on a variable
 *
 * @param variable Partition symbol to constrain
 * @param axes List of dimensions to broadcast
 *
 * @return Broadcast constraint
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

}  // namespace legate
