/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/type_traits.h>

#include <argparse/argparse.hpp>

#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

namespace legate::detail {

/**
 * @brief An object that models a "scaled" value.
 *
 * Used for example to model megabyte values more easily. Instead of storing 10MB in its direct
 * form (10,000,000), it stores "10" and "1,000,000".
 */
template <typename T>
class Scaled {
 public:
  using value_type = T;

  static_assert(std::is_integral_v<T>);

  /**
   * @brief Construct a `Scaled`.
   *
   * @param value The unscaled value (e.g. 10).
   * @param scale The scale to apply (e.g. 1,000,000).
   */
  Scaled(T value, type_identity_t<T> scale);

  /**
   * @return The unscaled value.
   */
  [[nodiscard]] T unscaled_value() const;

  /**
   * @return A mutable reference to the unscaled value.
   */
  [[nodiscard]] T& unscaled_value_mut();

  /**
   * @return The full, scaled value. Equivalent to `unscaled_value() * scale()`.
   */
  [[nodiscard]] T scaled_value() const;

  /**
   * @return The scaling factor.
   */
  [[nodiscard]] T scale() const;

 private:
  T value_{};
  T scale_{};
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Scaled<T>& arg);

// ==========================================================================================

/**
 * @brief An object containing a command-line argument.
 *
 * @tparam The type of the underlying value.
 */
template <typename T>
class Argument {
 public:
  using value_type = T;

  Argument() = delete;

  /**
   * @brief Construct an `Argument`.
   *
   * If the command-line argument was not set by the user, the value of `init` is used.
   *
   * @param parser The argument parser instance that created this `Argument`.
   * @param flag The command-line flag for the argument.
   * @param init The initial value for the `Argument`.
   */
  Argument(std::shared_ptr<argparse::ArgumentParser> parser, std::string flag, T init);

  /**
   * @return A reference to the `argparse::Argument` underpinning this object.
   *
   * This routine may be used to perform any other actions (like calling `hidden()`, or
   * `choices()`) not directly exposed by this object.
   */
  [[nodiscard]] argparse::Argument& argparse_argument();

  /**
   * @return The command-line flag for this argument.
   */
  [[nodiscard]] std::string_view flag() const;

  /**
   * @return A reference to the value of the argument.
   */
  [[nodiscard]] const T& value() const;

  /**
   * @return A mutable reference to the value of the argument.
   */
  [[nodiscard]] T& value_mut();

  /**
   * @return `true` if the value was set by the user on the command-line, `false` otherwise.
   */
  [[nodiscard]] bool was_set() const;

  /**
   * @brief Register an action to be taken if the user sets the flag on the command-line.
   *
   * The action is only taken if the flag is detected on the command-line. If the user does not
   * pass the flag, then no action is performed.
   *
   * The signature of `action` must be `T(std::string_view value, Argument<T> *arg)` where
   * `value` is the string value of the argument, and `arg` is a mutable pointer to this
   * object. The function must return the value that the argument should take. If the function
   * does not modify the argument, it may simply return `arg->value()`.
   *
   * It is possible to call this routine multiple times. Each additional action is invoked in
   * the order in which they were registered.
   *
   * This is overloaded with & because it passes a reference to this to the action, and so
   * should only be called on l-value type objects.
   *
   * @param action The callable action to take.
   *
   * @return A reference to this.
   */
  template <typename F>
  Argument& action(F&& action) &;

 private:
  std::shared_ptr<argparse::ArgumentParser> parser_{};
  std::string flag_{};
  T value_{};
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Argument<T>& arg);

}  // namespace legate::detail

#include <legate/runtime/detail/argument_parsing/argument.inl>
