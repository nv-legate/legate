/* Copyright 2021 NVIDIA Corporation
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

#include <map>
#include <memory>
#include <vector>

#include "core/utilities/tuple.h"

/** @defgroup partitioning Partitioning
 */

/**
 * @file
 * @brief Class definitions for partitioning constraint language
 */

namespace legate::detail {
class Operation;
class Strategy;
}  // namespace legate::detail

namespace legate {

class Alignment;
class Broadcast;
class Constraint;
class ImageConstraint;
class Literal;
class Partition;
class Variable;

/**
 * @ingroup partitioning
 * @brief A base class for expressions
 */
struct Expr {
  enum class Kind : int32_t {
    LITERAL  = 0,
    VARIABLE = 1,
  };

  virtual ~Expr() {}

  virtual void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const = 0;

  /**
   * @brief Indicates whether the expression is 'closed', i.e., free of any variables
   *
   * @return true Expression is closed
   * @return false Expression is not closed
   */
  virtual bool closed() const = 0;
  /**
   * @brief Converts the expressions to a human-readable string
   *
   * @return Expression in a string
   */
  virtual std::string to_string() const = 0;

  /**
   * @brief Returns the expression kind
   *
   * @return Expression kind
   */
  virtual Kind kind() const                   = 0;
  virtual const Literal* as_literal() const   = 0;
  virtual const Variable* as_variable() const = 0;
};

/**
 * @ingroup partitioning
 * @brief A class for literals
 */
class Literal : public Expr {
 public:
  Literal(const std::shared_ptr<Partition>& partition);

 public:
  Literal(const Literal&)            = default;
  Literal& operator=(const Literal&) = default;

 public:
  void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const override;

 public:
  bool closed() const override { return true; }
  std::string to_string() const override;

 public:
  Kind kind() const override { return Kind::LITERAL; }
  const Literal* as_literal() const override { return this; }
  const Variable* as_variable() const override { return nullptr; }

 public:
  std::shared_ptr<Partition> partition() const { return partition_; }

 private:
  std::shared_ptr<Partition> partition_;
};

/**
 * @ingroup partitioning
 * @brief Class for partition symbols
 */
class Variable : public Expr {
 public:
  Variable(const detail::Operation* op, int32_t id);

 public:
  Variable(const Variable&)            = default;
  Variable& operator=(const Variable&) = default;

 public:
  friend bool operator==(const Variable& lhs, const Variable& rhs);
  friend bool operator<(const Variable& lhs, const Variable& rhs);

 public:
  void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const override;

 public:
  bool closed() const override { return false; }
  std::string to_string() const override;

 public:
  Kind kind() const override { return Kind::VARIABLE; }
  const Literal* as_literal() const override { return nullptr; }
  const Variable* as_variable() const override { return this; }

 public:
  const detail::Operation* operation() const { return op_; }

 private:
  const detail::Operation* op_;
  int32_t id_;
};

/**
 * @ingroup partitioning
 * @brief A base class for partitioning constraints
 */
struct Constraint {
  enum class Kind : int32_t {
    ALIGNMENT = 0,
    BROADCAST = 1,
    IMAGE     = 2,
  };

  virtual ~Constraint() {}

  virtual void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const = 0;

  /**
   * @brief Converts the constraint to a human-readable string
   *
   * @return Constraint in a string
   */
  virtual std::string to_string() const = 0;

  /**
   * @brief Returns the constraint kind
   *
   * @return Constraint kind
   */
  virtual Kind kind() const = 0;

  virtual void validate() const = 0;

  virtual const Alignment* as_alignment() const              = 0;
  virtual const Broadcast* as_broadcast() const              = 0;
  virtual const ImageConstraint* as_image_constraint() const = 0;
};

/**
 * @ingroup partitioning
 * @brief A class for alignment constraints
 *
 * An alignment constraint on stores indicates that the stores should be partitioned in the same
 * way. If the stores referred to by an alignment constraint have different shapes, an
 * `std::invalid_argument` exception will be raised when the partitioner solves the constraint.
 */
class Alignment : public Constraint {
 public:
  Alignment(std::unique_ptr<Variable>&& lhs, std::unique_ptr<Variable>&& rhs);

 public:
  Kind kind() const override { return Kind::ALIGNMENT; }

 public:
  void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const override;

 public:
  void validate() const;

 public:
  std::string to_string() const override;

 public:
  const Alignment* as_alignment() const override { return this; }
  const Broadcast* as_broadcast() const override { return nullptr; }
  const ImageConstraint* as_image_constraint() const override { return nullptr; }

 public:
  /**
   * @brief Returns the LHS of the alignment constraint
   *
   * @return Variable
   */
  const Variable* lhs() const { return lhs_.get(); }
  /**
   * @brief Returns the RHS of the alignment constraint
   *
   * @return Variable
   */
  const Variable* rhs() const { return rhs_.get(); }

 public:
  bool is_trivial() const { return *lhs_ == *rhs_; }

 private:
  std::unique_ptr<Variable> lhs_;
  std::unique_ptr<Variable> rhs_;
};

/**
 * @ingroup partitioning
 * @brief A class for broadcast constraints
 *
 * A broadcast constraint on a store indicates that some or all dimensions of the store should not
 * be partitioned. The dimensions to broadcast must be specified by the constraint as well. If any
 * of the dimension names is invalid, an `std::invalid_argument` exception will be raised when the
 * auto-partitioner solves the constraint.
 */
class Broadcast : public Constraint {
 public:
  Broadcast(std::unique_ptr<Variable> variable, tuple<int32_t>&& axes);

 public:
  Kind kind() const override { return Kind::BROADCAST; }

 public:
  void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const override;

 public:
  void validate() const;

 public:
  std::string to_string() const override;

 public:
  const Alignment* as_alignment() const override { return nullptr; }
  const Broadcast* as_broadcast() const override { return this; }
  const ImageConstraint* as_image_constraint() const override { return nullptr; }

 public:
  /**
   * @brief Returns the partition symbol to which this constraint is applied
   *
   * @return Partition symbol
   */
  const Variable* variable() const { return variable_.get(); }
  /**
   * @brief Returns the list of axes to broadcast
   *
   * @return Tuple of integers
   */
  const tuple<int32_t>& axes() const { return axes_; }

 private:
  std::unique_ptr<Variable> variable_;
  tuple<int32_t> axes_;
};

/**
 * @ingroup partitioning
 * @brief A class for image constraints
 *
 * This constraint tells the partitioner that `var_range_` should be derived by collecting the image
 * of a "function", a store that contains either points or rects. `var_function_` is the partition
 * of the function that the partitioner should use in the image partitioning.
 */
class ImageConstraint : public Constraint {
 public:
  ImageConstraint(std::unique_ptr<Variable> var_function, std::unique_ptr<Variable> var_range);

 public:
  Kind kind() const override { return Kind::IMAGE; }

 public:
  void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const override;

 public:
  void validate() const;

 public:
  std::string to_string() const override;

 public:
  const Alignment* as_alignment() const override { return nullptr; }
  const Broadcast* as_broadcast() const override { return nullptr; }
  const ImageConstraint* as_image_constraint() const override { return this; }

 public:
  const Variable* var_function() const { return var_function_.get(); }
  const Variable* var_range() const { return var_range_.get(); }

 public:
  std::unique_ptr<Partition> resolve(const detail::Strategy& strategy) const;

 private:
  std::unique_ptr<Variable> var_function_;
  std::unique_ptr<Variable> var_range_;
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
std::unique_ptr<Alignment> align(const Variable* lhs, const Variable* rhs);

/**
 * @ingroup partitioning
 * @brief Creates a broadcast constraint on a variable
 *
 * @param variable Partition symbol to constrain
 * @param axes List of dimensions to broadcast
 *
 * @return Broadcast constraint
 */
std::unique_ptr<Broadcast> broadcast(const Variable* variable, const tuple<int32_t>& axes);

/**
 * @ingroup partitioning
 * @brief Creates a broadcast constraint on a variable
 *
 * @param variable Partition symbol to constrain
 * @param axes List of dimensions to broadcast
 *
 * @return Broadcast constraint
 */
std::unique_ptr<Broadcast> broadcast(const Variable* variable, tuple<int32_t>&& axes);

/**
 * @ingroup partitioning
 * @brief Creates an image constraint between partitions.
 *
 * @param var_function Partition symbol for the function store
 * @param var_range Partition symbol of the store whose partition should be derived from the image
 *
 * @return Broadcast constraint
 */
std::unique_ptr<ImageConstraint> image(const Variable* var_function, const Variable* var_range);

}  // namespace legate
