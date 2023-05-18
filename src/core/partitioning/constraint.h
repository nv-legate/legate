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

namespace legate {

struct Alignment;
struct Broadcast;
struct Constraint;
struct Literal;
struct Operation;
struct Partition;
struct Variable;

struct Expr {
 public:
  virtual ~Expr() {}

 public:
  virtual void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const = 0;

 public:
  virtual bool closed() const           = 0;
  virtual std::string to_string() const = 0;

 public:
  virtual const Literal* as_literal() const   = 0;
  virtual const Variable* as_variable() const = 0;
};

struct Literal : public Expr {
 public:
  Literal(const std::shared_ptr<Partition>& partition);

 public:
  Literal(const Literal&)            = default;
  Literal& operator=(const Literal&) = default;

 public:
  virtual void find_partition_symbols(
    std::vector<const Variable*>& partition_symbols) const override;

 public:
  virtual bool closed() const override { return true; }
  virtual std::string to_string() const override;

 public:
  virtual const Literal* as_literal() const override { return this; }
  virtual const Variable* as_variable() const override { return nullptr; }

 public:
  std::shared_ptr<Partition> partition() const { return partition_; }

 private:
  std::shared_ptr<Partition> partition_;
};

struct Variable : public Expr {
 public:
  Variable(const Operation* op, int32_t id);

 public:
  Variable(const Variable&)            = default;
  Variable& operator=(const Variable&) = default;

 public:
  friend bool operator==(const Variable& lhs, const Variable& rhs);
  friend bool operator<(const Variable& lhs, const Variable& rhs);

 public:
  virtual void find_partition_symbols(
    std::vector<const Variable*>& partition_symbols) const override;

 public:
  virtual bool closed() const override { return false; }
  virtual std::string to_string() const override;

 public:
  virtual const Literal* as_literal() const override { return nullptr; }
  virtual const Variable* as_variable() const override { return this; }

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
  };

  virtual ~Constraint() {}

  virtual Kind kind() const = 0;

  virtual void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const = 0;

  virtual std::string to_string() const = 0;

  virtual const Alignment* as_alignment() const = 0;
  virtual const Broadcast* as_broadcast() const = 0;
};

// Constraint AST nodes own their child nodes
class Alignment : public Constraint {
 public:
  Alignment(std::unique_ptr<Expr>&& lhs, std::unique_ptr<Expr>&& rhs);

 public:
  Kind kind() const override { return Kind::ALIGNMENT; }

 public:
  void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const override;

 public:
  std::string to_string() const override;

 public:
  const Alignment* as_alignment() const override { return this; }
  const Broadcast* as_broadcast() const override { return nullptr; }

 public:
  const Expr* lhs() const { return lhs_.get(); }
  const Expr* rhs() const { return rhs_.get(); }

 private:
  std::unique_ptr<Expr> lhs_;
  std::unique_ptr<Expr> rhs_;
};

class Broadcast : public Constraint {
 public:
  Broadcast(std::unique_ptr<Variable> variable, tuple<int32_t>&& axes);

 public:
  Kind kind() const override { return Kind::BROADCAST; }

 public:
  void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const override;

 public:
  std::string to_string() const override;

 public:
  const Alignment* as_alignment() const override { return nullptr; }
  const Broadcast* as_broadcast() const override { return this; }

 public:
  const Variable* variable() const { return variable_.get(); }
  const tuple<int32_t>& axes() const { return axes_; }

 private:
  std::unique_ptr<Variable> variable_;
  tuple<int32_t> axes_;
};

std::unique_ptr<Constraint> align(const Variable* lhs, const Variable* rhs);

std::unique_ptr<Constraint> broadcast(const Variable* variable, const tuple<int32_t>& axes);

std::unique_ptr<Constraint> broadcast(const Variable* variable, tuple<int32_t>&& axes);

}  // namespace legate
