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

#include "core/utilities/tuple.h"
#include "core/utilities/typedefs.h"

#include <functional>
#include <iosfwd>
#include <tuple>

/**
 * @file
 * @brief Definitions for legate::SymbolicExpr and constructors for symbolic expressions
 */

namespace legate {

/**
 * @ingroup op
 * @brief A class that symbolically represents coordinates.
 *
 * A @f$\mathtt{SymbolicExpr}(i, w, c)@f$ object denotes an expression @f$ w \cdot \mathit{dim}_i +
 * c
 * @f$, where @f$ \mathit{dim}_i @f$ corresponds to the coordinate of the @f$i@f$-th dimension. A
 * special case is when @f$i@f$ is @f$-1@f$, which means the expression denotes a constant
 * @f$c@f$.
 */
class SymbolicExpr {
 public:
  static constexpr std::uint32_t UNSET = -1U;

  SymbolicExpr() = default;

  SymbolicExpr(std::uint32_t dim, std::int32_t weight, std::int32_t offset = 0);

  explicit SymbolicExpr(std::uint32_t dim);

  /**
   * @brief Returns the dimension index of this expression
   *
   * @return Dimension index
   */
  [[nodiscard]] std::uint32_t dim() const;
  /**
   * @brief Returns the weight for the coordinates
   *
   * @return Weight value
   */
  [[nodiscard]] std::int32_t weight() const;
  /**
   * @brief Returns the offset of the expression
   *
   * @return Offset
   */
  [[nodiscard]] std::int32_t offset() const;

  /**
   * @brief Indicates if the expression denotes an identity mapping for the given dimension
   *
   * @param dim The dimension for which the identity mapping is checked
   *
   * @return true The expression denotes an identity mapping
   * @return false The expression does not denote an identity mapping
   */
  [[nodiscard]] bool is_identity(std::uint32_t dim) const;
  /**
   * @brief Indicates if the expression denotes a constant
   *
   * @return true The expression denotes a constant
   * @return false The expression does not denote a constant
   */
  [[nodiscard]] bool is_constant() const;

  bool operator==(const SymbolicExpr& other) const;
  bool operator<(const SymbolicExpr& other) const;

  SymbolicExpr operator*(std::int32_t other) const;
  SymbolicExpr operator+(std::int32_t other) const;

  [[nodiscard]] std::string to_string() const;

  [[nodiscard]] std::size_t hash() const;

 private:
  std::uint32_t dim_{UNSET};
  std::int32_t weight_{1};
  std::int32_t offset_{};
};

std::ostream& operator<<(std::ostream& out, const SymbolicExpr& expr);

/**
 * @ingroup op
 * @brief A symbolic representation of points
 *
 * Symbolic points are used to capture mappings between points in different
 * domains in a concise way. Each element of a symbolic point is a
 * `SymbolicExpr` symbolically representing the coordinate of that dimension. A
 * `ManualTask` can optionally pass for its logical store partition argument a
 * symbolic point that describes a mapping from points in the launch domain to
 * sub-stores in the partition.
 */
using SymbolicPoint = tuple<SymbolicExpr>;

/**
 * @ingroup op
 * @brief Constructs a `SymbolicExpr` representing coordinates of a dimension
 *
 * @param dim The dimension index
 *
 * @return A symbolic expression for the given dimension
 */
[[nodiscard]] SymbolicExpr dimension(std::uint32_t dim);

/**
 * @ingroup op
 * @brief Constructs a `SymbolicExpr` representing a constant value.
 *
 * @param value The constant value to embed
 *
 * @return A symbolic expression for the given constant
 */
[[nodiscard]] SymbolicExpr constant(std::int32_t value);

}  // namespace legate

#include "core/operation/projection.inl"
