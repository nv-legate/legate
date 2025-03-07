/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/shared_ptr.h>

#include <cstdint>

/**
 * @file
 * @brief Class definitions for proxy constraint objects.
 */

namespace legate::detail {

class ProxyConstraint;

}  // namespace legate::detail

namespace legate {

/**
 * @addtogroup partitioning
 * @{
 */

/**
 * @brief An object that models a specific array argument to a task.
 */
class ProxyArrayArgument {
 public:
  /**
   * @brief The kind of argument.
   */
  enum class Kind : std::uint8_t {
    INPUT,
    OUTPUT,
    REDUCTION,
  };

  [[nodiscard]] constexpr bool operator==(const ProxyArrayArgument& rhs) const noexcept;
  [[nodiscard]] constexpr bool operator!=(const ProxyArrayArgument& rhs) const noexcept;

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
// "detail" (used later) doesn't name a type or namespace, because now they read it as
// "legate::proxy::detail" instead of "legate::detail"
namespace proxy_detail {

template <typename T, ProxyArrayArgument::Kind KIND>
class TaskArgsBase {
 public:
  [[nodiscard]] constexpr bool operator==(const TaskArgsBase& rhs) const noexcept;
  [[nodiscard]] constexpr bool operator!=(const TaskArgsBase& rhs) const noexcept;

  [[nodiscard]] constexpr ProxyArrayArgument operator[](std::uint32_t index) const noexcept;

 private:
  friend T;

  constexpr TaskArgsBase() = default;
};

}  // namespace proxy_detail

/**
 * @brief A class that models the input arguments to a task.
 */
class ProxyInputArguments
  : public proxy_detail::TaskArgsBase<ProxyInputArguments, ProxyArrayArgument::Kind::INPUT> {
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

// ==========================================================================================

/**
 * @brief A class that models the output arguments to a task.
 */
class ProxyOutputArguments
  : public proxy_detail::TaskArgsBase<ProxyOutputArguments, ProxyArrayArgument::Kind::OUTPUT> {
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

// ==========================================================================================

/**
 * @brief A class that models the reduction arguments to a task.
 */
class ProxyReductionArguments
  : public proxy_detail::TaskArgsBase<ProxyReductionArguments,
                                      ProxyArrayArgument::Kind::REDUCTION> {
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

// ==========================================================================================

/**
 * @brief The base proxy constraint class.
 */
class ProxyConstraint {
 public:
  ProxyConstraint()                                           = LEGATE_DEFAULT_WHEN_CYTHON;
  ProxyConstraint(const ProxyConstraint&) noexcept            = default;
  ProxyConstraint& operator=(const ProxyConstraint&) noexcept = default;
  ProxyConstraint(ProxyConstraint&&) noexcept                 = default;
  ProxyConstraint& operator=(ProxyConstraint&&) noexcept      = default;
  ~ProxyConstraint();

  /**
   * @brief Construct a proxy constraint.
   *
   * @param impl The pointer to the private implementation.
   */
  explicit ProxyConstraint(SharedPtr<detail::ProxyConstraint> impl);

  /**
   * @return The pointer to the private implementation.
   */
  [[nodiscard]] const SharedPtr<detail::ProxyConstraint>& impl() const;

 private:
  SharedPtr<detail::ProxyConstraint> impl_;
};

/** @} */

}  // namespace legate

namespace legate::proxy {

/**
 * @addtogroup partitioning
 * @{
 */

/**
 * @brief A proxy object that models the input arguments to a task as whole.
 *
 * @see ProxyInputArguments
 */
inline constexpr ProxyInputArguments inputs{};  // NOLINT(readability-identifier-naming)

/**
 * @brief A proxy object that models the output arguments to a task as whole.
 *
 * @see ProxyOutputArguments
 */
inline constexpr ProxyOutputArguments outputs{};  // NOLINT(readability-identifier-naming)

/**
 * @brief A proxy object that models the reduction arguments to a task as whole.
 *
 * @see ProxyReductionArguments
 */
inline constexpr ProxyReductionArguments reductions{};  // NOLINT(readability-identifier-naming)

/** @} */

}  // namespace legate::proxy

#include <legate/partitioning/proxy.inl>
