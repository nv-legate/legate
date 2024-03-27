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

#include "core/utilities/defined.h"
#include "core/utilities/detail/compressed_pair.h"

#include <type_traits>

/** @addtogroup util
 *  @{
 */

/**
 * @file
 * @brief Definitions of utilities relating to normal and exceptional scope exit.
 */

namespace legate {

/**
 * @brief A simple wrapper around a callable that automatically executes the callable on
 * exiting the scope.
 *
 * @tparam F The type of the callable to execute.
 */
template <typename F>
class ScopeGuard {
 public:
  /**
   * The type of callable stored within the ScopeGuard.
   */
  using value_type = F;

  static_assert(std::is_nothrow_move_constructible_v<value_type>);
  static_assert(std::is_nothrow_invocable_v<value_type>);

  ScopeGuard()                             = delete;
  ScopeGuard(const ScopeGuard&)            = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;

  /**
   * @brief Construct a ScopeGuard.
   *
   * @param fn The function to execute.
   * @param enabled Whether the ScopeGuard should start in the "enabled" state.
   *
   * On destruction, a ScopeGuard will execute \p fn if and only if it is in the enabled
   * state. \p fn will be invoked with no arguments, and any return value discarded. \p fn must
   * be no-throw move-constructible, and must not throw any exceptions when invoked.
   *
   * @see ScopeGuard::enable()
   * @see ScopeGuard::disable()
   * @see ScopeGuard::enabled()
   * @see ScopeFail
   */
  explicit ScopeGuard(value_type&& fn, bool enabled = true) noexcept;

  /**
   * @brief Move-construct a ScopeGuard.
   *
   * @param other The ScopeGuard to move from.
   *
   * \p other will be left in the "disabled" state, and will not execute its held functor upon
   * destruction. Furthermore, the held functor is moved into the receiving ScopeGuard, so \p
   * other's functor may be in an indeterminate state. It is therefore not advised to re-enable
   * \p other.
   */
  ScopeGuard(ScopeGuard&& other) noexcept;

  /**
   * @brief Construct a ScopeGuard via move-assignment
   *
   * @param other The ScopeGuard to move from.
   * @return A reference to `this`.
   *
   * This routine has no effect if \p other and `this` are the same.
   *
   * \p other will be left in the "disabled" state, and will not execute its held functor upon
   * destruction. Furthermore, the held functor is moved into the receiving ScopeGuard, so \p
   * other's functor may be in an indeterminate state. It is therefore not advised to re-enable
   * \p other.
   */
  ScopeGuard& operator=(ScopeGuard&& other) noexcept;

  /**
   * @brief Destroy a ScopeGuard.
   *
   * If the ScopeGuard is currently in the enabled state, executes the held functor, otherwise
   * does nothing.
   */
  ~ScopeGuard() noexcept;

  /**
   * @brief Query a ScopeGuard's state.
   *
   * @return true if the ScopeGuard is enabled, false otherwise.
   *
   * @see ScopeGuard::enable()
   * @see ScopeGuard::disable()
   */
  [[nodiscard]] bool enabled() const;

  /**
   * @brief Disable a ScopeGuard.
   *
   * This routine prevents a ScopeGuard from executing its held functor on destruction. On
   * return, ScopeGuard::enabled() will return false.
   *
   * Calling this routine on an already disabled ScopeGuard has no effect.
   *
   * @see ScopeGuard::enable()
   */
  void disable();

  /**
   * @brief Enable a ScopeGuard.
   *
   * This routine makes a ScopeGuard execute its held functor on destruction. On return,
   * ScopeGuard::enabled() will return true.
   *
   * Calling this routine on an already enabled ScopeGuard has no effect.
   *
   * @see ScopeGuard::disable()
   */
  void enable();

 private:
  [[nodiscard]] bool& enabled_();
  [[nodiscard]] bool enabled_() const;

  [[nodiscard]] value_type& func_();
  [[nodiscard]] const value_type& func_() const;

  detail::compressed_pair<value_type, bool> pair_{};
};

/**
 * @brief Create a ScopeGuard from a given functor.
 *
 * @param fn The functor to create the ScopeGuard with.
 *
 * @returns The constructed ScopeGuard
 *
 * @tparam The type of \p fn, usually inferred from the argument itself.
 *
 * @see ScopeGuard
 */
template <typename F>
[[nodiscard]] ScopeGuard<F> make_scope_guard(F&& fn) noexcept;

// Hide the initializer for this, the user doesn't care how the sausage is made...
/**
 * \hideinitializer
 * @brief Construct an unnamed ScopeGuard from the contents of the macro arguments.
 *
 * @param ... The body of the constructed ScopeGuard.
 *
 * It is impossible to enable or disable the ScopeGuard constructed by this macro.
 *
 * This macro is useful if the user need only define some action to be executed on scope exit,
 * but doesn't care to name the ScopeGuard and/or has no need to enable/disable it after
 * construction.
 *
 * For example:
 *
 * @code
 * int *mem = std::malloc(10 * sizeof(int));
 *
 * LEGATE_SCOPE_GUARD(std::free(mem));
 * // use mem...
 * // scope exits, and mem is free'd.
 * @endcode
 *
 * Multi-line statements are also supported:
 *
 * @code
 * int *mem = std::malloc(10 * sizeof(int));
 *
 * LEGATE_SCOPE_GUARD(
 *   if (frobnicate()) {
 *     std::free(mem);
 *   }
 * );
 * // use mem...
 * // scope exits, and mem is free'd depending on return value of frobnicate()
 * @endcode
 *
 * If the body of the guard should only be executed on failure, use LEGATE_SCOPE_FAIL instead.
 *
 * @see ScopeGuard
 * @see LEGATE_SCOPE_FAIL
 */
#define LEGATE_SCOPE_GUARD(...)                                   \
  const auto LegateConcat(__legate_core_scope_guard_, __LINE__) = \
    ::legate::make_scope_guard([&]() noexcept { __VA_ARGS__; })

/**
 * @brief Similar to ScopeGuard, except that the callable is only executed if the scope is
 * exited due to an exception.
 *
 * @tparam F The type of the callable to execute.
 */
template <typename F>
class ScopeFail {
 public:
  /**
   * The type of callable stored within the ScopeFail.
   */
  using value_type = F;

  static_assert(std::is_nothrow_move_constructible_v<value_type>);
  static_assert(std::is_nothrow_invocable_v<value_type>);

  /**
   * @brief Construct a ScopeFail.
   *
   * @param fn The function to execute.
   *
   * On destruction, a ScopeFail will execute \p fn if and only if the scope is being exited
   * due to an uncaught exception. Therefore, unlike ScopeGuard, it is not possible to
   * "disable" a ScopeFail.
   *
   * \p fn will be invoked with no arguments, and any return value discarded. \p fn must be
   * no-throw move-constructible, and must not throw any exceptions when invoked.
   *
   * @see ScopeGuard
   */
  explicit ScopeFail(value_type&& fn) noexcept;

  // ScopeFails are neither default constructible, copy, or move constructible due to the
  // inability to distinguish a moved-from guard from a disabled guard. Not that I suspect this
  // class will ever be moved anyways...
  ScopeFail()                                      = delete;
  ScopeFail(const ScopeFail&)                      = delete;
  ScopeFail& operator=(const ScopeFail&)           = delete;
  ScopeFail(ScopeFail&& other) noexcept            = delete;
  ScopeFail& operator=(ScopeFail&& other) noexcept = delete;

  /**
   * @brief Destroy a ScopeFail.
   *
   * If the ScopeFail is being destroyed due to the result of exception-related stack
   * unwinding, then the held functor is executed, otherwise has no effect.
   */
  ~ScopeFail() noexcept;

 private:
  [[nodiscard]] int exn_count_() const;

  ScopeGuard<value_type> guard_{};
  int exn_cnt_{};
};

/**
 * @brief Create a ScopeFail from a given functor.
 *
 * @param fn The functor to create the ScopeFail with.
 *
 * @returns The constructed ScopeFail
 *
 * @tparam The type of \p fn, usually inferred from the argument itself.
 *
 * @see ScopeFail
 */
template <typename F>
[[nodiscard]] ScopeFail<F> make_scope_fail(F&& fn) noexcept;

// Hide the initializer for this, the user doesn't care how the sausage is made...
/**
 * \hideinitializer
 * @brief Construct an unnamed ScopeFail from the contents of the macro arguments.
 *
 * @param ... The body of the constructed ScopeGuard.
 *
 * This macro behaves identically to \ref LEGATE_SCOPE_GUARD, except that it creates a
 * ScopeFail instead of a \ref ScopeGuard. Please refer to its documentation for further discussion.
 *
 * @see ScopeFail
 * @see LEGATE_SCOPE_GUARD
 */
#define LEGATE_SCOPE_FAIL(...)                                   \
  const auto LegateConcat(__legate_core_scope_fail_, __LINE__) = \
    ::legate::make_scope_fail([&]() noexcept { __VA_ARGS__; })

}  // namespace legate

/** @} */  // end of group util

#include "core/utilities/scope_guard.inl"
