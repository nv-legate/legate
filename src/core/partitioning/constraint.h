/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include <memory>

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
class Constraint;
class Variable;
}  // namespace detail

/**
 * @ingroup partitioning
 * @brief Class for partition symbols
 */
class Variable {
 public:
  std::string to_string() const;

 public:
  Variable(const detail::Variable* impl);
  ~Variable();

 public:
  Variable(const Variable&);
  Variable& operator=(const Variable&);

 private:
  Variable(Variable&&)            = delete;
  Variable& operator=(Variable&&) = delete;

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
  ~Constraint();

 public:
  Constraint(Constraint&&);
  Constraint& operator=(Constraint&&);

 private:
  Constraint(const Constraint&)            = delete;
  Constraint& operator=(const Constraint&) = delete;

 private:
  friend class AutoTask;
  detail::Constraint* release();

 private:
  detail::Constraint* impl_{nullptr};
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
