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

#include <map>
#include <memory>
#include <vector>

#include "core/utilities/tuple.h"

namespace legate {
class Partition;
}  // namespace legate

namespace legate::detail {
class Operation;
class Strategy;

class Alignment;
class Broadcast;
class Constraint;
class ImageConstraint;
class Literal;
class Variable;

struct Expr {
  enum class Kind : int32_t {
    LITERAL  = 0,
    VARIABLE = 1,
  };
  virtual ~Expr() {}
  virtual void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const = 0;
  virtual bool closed() const                                                                = 0;
  virtual std::string to_string() const                                                      = 0;
  virtual Kind kind() const                                                                  = 0;
  virtual const Literal* as_literal() const                                                  = 0;
  virtual const Variable* as_variable() const                                                = 0;
};

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

class Variable : public Expr {
 public:
  Variable(const Operation* op, int32_t id);

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
  const Operation* operation() const { return op_; }

 private:
  const Operation* op_;
  int32_t id_;
};

struct Constraint {
  enum class Kind : int32_t {
    ALIGNMENT = 0,
    BROADCAST = 1,
    IMAGE     = 2,
  };
  virtual ~Constraint() {}
  virtual void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const = 0;
  virtual std::string to_string() const                                                      = 0;
  virtual Kind kind() const                                                                  = 0;
  virtual void validate() const                                                              = 0;
  virtual const Alignment* as_alignment() const                                              = 0;
  virtual const Broadcast* as_broadcast() const                                              = 0;
  virtual const ImageConstraint* as_image_constraint() const                                 = 0;
};

class Alignment : public Constraint {
 public:
  Alignment(const Variable* lhs, const Variable* rhs);

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
  const Variable* lhs() const { return lhs_; }
  const Variable* rhs() const { return rhs_; }

 public:
  bool is_trivial() const { return *lhs_ == *rhs_; }

 private:
  const Variable* lhs_;
  const Variable* rhs_;
};

class Broadcast : public Constraint {
 public:
  Broadcast(const Variable* variable, const tuple<int32_t>& axes);

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
  const Variable* variable() const { return variable_; }
  const tuple<int32_t>& axes() const { return axes_; }

 private:
  const Variable* variable_;
  tuple<int32_t> axes_;
};

class ImageConstraint : public Constraint {
 public:
  ImageConstraint(const Variable* var_function, const Variable* var_range);

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
  const Variable* var_function() const { return var_function_; }
  const Variable* var_range() const { return var_range_; }

 public:
  std::unique_ptr<Partition> resolve(const Strategy& strategy) const;

 private:
  const Variable* var_function_;
  const Variable* var_range_;
};

std::unique_ptr<Alignment> align(const Variable* lhs, const Variable* rhs);

std::unique_ptr<Broadcast> broadcast(const Variable* variable, const tuple<int32_t>& axes);

std::unique_ptr<ImageConstraint> image(const Variable* var_function, const Variable* var_range);

}  // namespace legate::detail
