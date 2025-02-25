/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/shared_ptr.h>

#include <cstdint>

/**
 * @file
 * @brief Class definitions for proxy contraint objects.
 */

namespace legate {

enum class ImageComputationHint : std::uint8_t;

}  // namespace legate

namespace legate::detail::proxy {

class Constraint;

}  // namespace legate::detail::proxy

namespace legate::proxy {

/**
 * @addtogroup partitioning
 * @{
 */

/**
 * @brief An object that models a specific array argument to a task.
 */
class ArrayArgument {
 public:
  /**
   * @brief The kind of argument.
   */
  enum class Kind : std::uint8_t {
    INPUT,
    OUTPUT,
    REDUCTION,
  };

  [[nodiscard]] constexpr bool operator==(const ArrayArgument& rhs) const noexcept;
  [[nodiscard]] constexpr bool operator!=(const ArrayArgument& rhs) const noexcept;

  /**
   * @brief The selected kind of the argument.
   */
  Kind kind{};

  /**
   * @brief The index into the argument list (as returned e.g. by `TaskContext::inputs()`)
   * corresponding to the argument.
   */
  std::uint32_t index{};
};

// ==========================================================================================

// Don't use namespace detail here because otherwise compilers complain that bare
// "detail::proxy" (used later) doesn't name a type or namespace, because now they read it as
// "legate::proxy::detail::proxy" instead of "legate::detail::proxy"
namespace proxy_detail {

template <typename T, ArrayArgument::Kind KIND>
class TaskArgsBase {
 public:
  [[nodiscard]] constexpr bool operator==(const TaskArgsBase& rhs) const noexcept;
  [[nodiscard]] constexpr bool operator!=(const TaskArgsBase& rhs) const noexcept;

  [[nodiscard]] constexpr ArrayArgument operator[](std::uint32_t index) const noexcept;

 private:
  friend T;

  constexpr TaskArgsBase() = default;
};

}  // namespace proxy_detail

/**
 * @brief A class that models the input arguments to a task.
 */
class InputArguments
  : public proxy_detail::TaskArgsBase<InputArguments, ArrayArgument::Kind::INPUT> {
 public:
  using TaskArgsBase::TaskArgsBase;

  /**
   * @brief Selects a specific argument from the input arguments.
   *
   * @param index The index into the array of arguments. Analogous to what is passed to
   * `TaskContext::input()`.
   *
   * @return A model of the selected input argument.
   */
  using TaskArgsBase::operator[];
};

/**
 * @brief A proxy object that models the input arguments to a task as whole.
 *
 * @see InputArguments
 */
inline constexpr InputArguments inputs{};  // NOLINT(readability-identifier-naming)

// ==========================================================================================

/**
 * @brief A class that models the output arguments to a task.
 */
class OutputArguments
  : public proxy_detail::TaskArgsBase<OutputArguments, ArrayArgument::Kind::OUTPUT> {
 public:
  using TaskArgsBase::TaskArgsBase;

  /**
   * @brief Selects a specific argument from the output arguments.
   *
   * @param index The index into the array of arguments. Analogous to what is passed to
   * `TaskContext::output()`.
   *
   * @return A model of the selected output argument.
   */
  using TaskArgsBase::operator[];
};

/**
 * @brief A proxy object that models the output arguments to a task as whole.
 *
 * @see OutputArguments
 */
inline constexpr OutputArguments outputs{};  // NOLINT(readability-identifier-naming)

// ==========================================================================================

/**
 * @brief A class that models the reduction arguments to a task.
 */
class ReductionArguments
  : public proxy_detail::TaskArgsBase<ReductionArguments, ArrayArgument::Kind::REDUCTION> {
 public:
  using TaskArgsBase::TaskArgsBase;

  /**
   * @brief Selects a specific argument from the reduction arguments.
   *
   * @param index The index into the array of arguments. Analogous to what is passed to
   * `TaskContext::reduction()`.
   *
   * @return A model of the selected reduction argument.
   */
  using TaskArgsBase::operator[];
};

/**
 * @brief A proxy object that models the reduction arguments to a task as whole.
 *
 * @see ReductionArguments
 */
inline constexpr ReductionArguments reductions{};  // NOLINT(readability-identifier-naming)

// ==========================================================================================

/**
 * @brief The base proxy constraint class.
 */
class Constraint {
 public:
  Constraint()                                      = LEGATE_DEFAULT_WHEN_CYTHON;
  Constraint(const Constraint&) noexcept            = default;
  Constraint& operator=(const Constraint&) noexcept = default;
  Constraint(Constraint&&) noexcept                 = default;
  Constraint& operator=(Constraint&&) noexcept      = default;
  ~Constraint();

  /**
   * @brief Construct a proxy constraint.
   *
   * @param impl The pointer to the private implementation.
   */
  explicit Constraint(SharedPtr<detail::proxy::Constraint> impl);

  /**
   * @return The pointer to the private implementation.
   */
  [[nodiscard]] const SharedPtr<detail::proxy::Constraint>& impl() const;

 private:
  SharedPtr<detail::proxy::Constraint> impl_;
};

/** @} */

}  // namespace legate::proxy

#include <legate/partitioning/proxy.inl>
