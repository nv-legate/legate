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

namespace legate {

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
 public:
  virtual ~Constraint() {}

 public:
  virtual void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const = 0;

 public:
  virtual std::string to_string() const = 0;
};

std::unique_ptr<Constraint> align(const Variable* lhs, const Variable* rhs);

}  // namespace legate
