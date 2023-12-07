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
#include "core/utilities/tuple.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace legate {
struct Partition;
}  // namespace legate

namespace legate::detail {
class Operation;
class Strategy;

class Alignment;
class BloatConstraint;
class Broadcast;
struct Constraint;
class ImageConstraint;
class Literal;
class ScaleConstraint;
class Variable;

struct Expr {
  enum class Kind : int32_t {
    LITERAL  = 0,
    VARIABLE = 1,
  };
  Expr()                           = default;
  virtual ~Expr()                  = default;
  Expr(const Expr&)                = default;
  Expr(Expr&&) noexcept            = default;
  Expr& operator=(const Expr&)     = default;
  Expr& operator=(Expr&&) noexcept = default;

  virtual void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const = 0;
  [[nodiscard]] virtual bool closed() const                                                  = 0;
  [[nodiscard]] virtual std::string to_string() const                                        = 0;
  [[nodiscard]] virtual Kind kind() const                                                    = 0;
  [[nodiscard]] virtual const Literal* as_literal() const                                    = 0;
  [[nodiscard]] virtual const Variable* as_variable() const                                  = 0;
};

class Literal final : public Expr {
 public:
  explicit Literal(std::shared_ptr<Partition> partition);

  void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const override;

  [[nodiscard]] bool closed() const override;
  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] Kind kind() const override;
  [[nodiscard]] const Literal* as_literal() const override;
  [[nodiscard]] const Variable* as_variable() const override;

  [[nodiscard]] const std::shared_ptr<Partition>& partition() const;

 private:
  std::shared_ptr<Partition> partition_{};
};

class Variable final : public Expr {
 public:
  Variable(const Operation* op, int32_t id);

  friend bool operator==(const Variable& lhs, const Variable& rhs);

  void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const override;

  [[nodiscard]] bool closed() const override;
  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] Kind kind() const override;
  [[nodiscard]] const Literal* as_literal() const override;
  [[nodiscard]] const Variable* as_variable() const override;

  [[nodiscard]] const Operation* operation() const;

  [[nodiscard]] size_t hash() const noexcept;

 private:
  const Operation* op_{};
  int32_t id_{};
};

struct Constraint {
  enum class Kind : int32_t {
    ALIGNMENT = 0,
    BROADCAST = 1,
    IMAGE     = 2,
    SCALE     = 3,
    BLOAT     = 4,
  };
  virtual ~Constraint() = default;
  virtual void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const = 0;
  [[nodiscard]] virtual std::string to_string() const                                        = 0;
  [[nodiscard]] virtual Kind kind() const                                                    = 0;
  virtual void validate() const                                                              = 0;
  [[nodiscard]] virtual const Alignment* as_alignment() const                                = 0;
  [[nodiscard]] virtual const Broadcast* as_broadcast() const                                = 0;
  [[nodiscard]] virtual const ImageConstraint* as_image_constraint() const                   = 0;
  [[nodiscard]] virtual const ScaleConstraint* as_scale_constraint() const                   = 0;
  [[nodiscard]] virtual const BloatConstraint* as_bloat_constraint() const                   = 0;
};

class Alignment final : public Constraint {
 public:
  Alignment(const Variable* lhs, const Variable* rhs);

  [[nodiscard]] Kind kind() const override;

  void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const override;

  void validate() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const Alignment* as_alignment() const override;
  [[nodiscard]] const Broadcast* as_broadcast() const override;
  [[nodiscard]] const ImageConstraint* as_image_constraint() const override;
  [[nodiscard]] const ScaleConstraint* as_scale_constraint() const override;
  [[nodiscard]] const BloatConstraint* as_bloat_constraint() const override;

  [[nodiscard]] const Variable* lhs() const;
  [[nodiscard]] const Variable* rhs() const;

  [[nodiscard]] bool is_trivial() const;

 private:
  const Variable* lhs_{};
  const Variable* rhs_{};
};

class Broadcast final : public Constraint {
 public:
  explicit Broadcast(const Variable* variable);

  Broadcast(const Variable* variable, tuple<int32_t> axes);

  [[nodiscard]] Kind kind() const override;

  void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const override;

  void validate() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const Alignment* as_alignment() const override;
  [[nodiscard]] const Broadcast* as_broadcast() const override;
  [[nodiscard]] const ImageConstraint* as_image_constraint() const override;
  [[nodiscard]] const ScaleConstraint* as_scale_constraint() const override;
  [[nodiscard]] const BloatConstraint* as_bloat_constraint() const override;

  [[nodiscard]] const Variable* variable() const;
  [[nodiscard]] const tuple<int32_t>& axes() const;

 private:
  const Variable* variable_{};
  // Broadcast all dimensions when empty
  tuple<int32_t> axes_{};
};

class ImageConstraint final : public Constraint {
 public:
  ImageConstraint(const Variable* var_function, const Variable* var_range);

  [[nodiscard]] Kind kind() const override;

  void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const override;

  void validate() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const Alignment* as_alignment() const override;
  [[nodiscard]] const Broadcast* as_broadcast() const override;
  [[nodiscard]] const ImageConstraint* as_image_constraint() const override;
  [[nodiscard]] const ScaleConstraint* as_scale_constraint() const override;
  [[nodiscard]] const BloatConstraint* as_bloat_constraint() const override;

  [[nodiscard]] const Variable* var_function() const;
  [[nodiscard]] const Variable* var_range() const;

  [[nodiscard]] std::shared_ptr<Partition> resolve(const Strategy& strategy) const;

 private:
  const Variable* var_function_{};
  const Variable* var_range_{};
};

class ScaleConstraint final : public Constraint {
 public:
  ScaleConstraint(Shape factors, const Variable* var_smaller, const Variable* var_bigger);

  [[nodiscard]] Kind kind() const override;

  void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const override;

  void validate() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const Alignment* as_alignment() const override;
  [[nodiscard]] const Broadcast* as_broadcast() const override;
  [[nodiscard]] const ImageConstraint* as_image_constraint() const override;
  [[nodiscard]] const ScaleConstraint* as_scale_constraint() const override;
  [[nodiscard]] const BloatConstraint* as_bloat_constraint() const override;

  [[nodiscard]] const Variable* var_smaller() const;
  [[nodiscard]] const Variable* var_bigger() const;

  [[nodiscard]] std::shared_ptr<Partition> resolve(const Strategy& strategy) const;

 private:
  Shape factors_{};
  const Variable* var_smaller_{};
  const Variable* var_bigger_{};
};

class BloatConstraint final : public Constraint {
 public:
  BloatConstraint(const Variable* var_source,
                  const Variable* var_bloat,
                  Shape low_offsets,
                  Shape high_offsets);

  [[nodiscard]] Kind kind() const override;

  void find_partition_symbols(std::vector<const Variable*>& partition_symbols) const override;

  void validate() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const Alignment* as_alignment() const override;
  [[nodiscard]] const Broadcast* as_broadcast() const override;
  [[nodiscard]] const ImageConstraint* as_image_constraint() const override;
  [[nodiscard]] const ScaleConstraint* as_scale_constraint() const override;
  [[nodiscard]] const BloatConstraint* as_bloat_constraint() const override;

  [[nodiscard]] const Variable* var_source() const;
  [[nodiscard]] const Variable* var_bloat() const;

  [[nodiscard]] std::shared_ptr<Partition> resolve(const Strategy& strategy) const;

 private:
  const Variable* var_source_{};
  const Variable* var_bloat_{};
  Shape low_offsets_{};
  Shape high_offsets_{};
};

[[nodiscard]] std::shared_ptr<Alignment> align(const Variable* lhs, const Variable* rhs);

[[nodiscard]] std::shared_ptr<Broadcast> broadcast(const Variable* variable);

[[nodiscard]] std::shared_ptr<Broadcast> broadcast(const Variable* variable,
                                                   const tuple<int32_t>& axes);

[[nodiscard]] std::shared_ptr<ImageConstraint> image(const Variable* var_function,
                                                     const Variable* var_range);

[[nodiscard]] std::shared_ptr<ScaleConstraint> scale(const Shape& factors,
                                                     const Variable* var_smaller,
                                                     const Variable* var_bigger);

[[nodiscard]] std::shared_ptr<BloatConstraint> bloat(const Variable* var_source,
                                                     const Variable* var_bloat,
                                                     const Shape& low_offsets,
                                                     const Shape& high_offsets);

}  // namespace legate::detail

#include "core/partitioning/detail/constraint.inl"
