/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/utilities/detail/string_utils.h>

#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace legate::detail {

template <typename T>
Scaled<T>::Scaled(T value, type_identity_t<T> scale, std::string_view unit)
  : value_{std::move(value)}, scale_{std::move(scale)}, unit_{unit}
{
}

template <typename T>
T Scaled<T>::unscaled_value() const
{
  return value_;
}

template <typename T>
T& Scaled<T>::unscaled_value_mut()
{
  return value_;
}

template <typename T>
T Scaled<T>::scaled_value() const
{
  return unscaled_value() * scale();
}

template <typename T>
T Scaled<T>::scale() const
{
  return scale_;
}

template <typename T>
std::string_view Scaled<T>::unit() const
{
  return unit_;
}

// ------------------------------------------------------------------------------------------

template <typename T>
std::ostream& operator<<(std::ostream& os, const Scaled<T>& arg)
{
  os << "Scaled(scale: " << arg.scale() << ", value: " << arg.unscaled_value() << ")";
  return os;
}

// ==========================================================================================

template <typename T>
Argument<T>::Argument(std::shared_ptr<argparse::ArgumentParser> parser, std::string flag, T init)
  : parser_{std::move(parser)}, flag_{std::move(flag)}, value_{std::move(init)}
{
}

template <typename T>
argparse::Argument& Argument<T>::argparse_argument()
{
  return (*parser_)[flag()];
}

template <typename T>
std::string_view Argument<T>::flag() const
{
  return flag_;
}

template <typename T>
std::string_view Argument<T>::name() const
{
  return string_remove_prefix(flag(), "--");
}

template <typename T>
const T& Argument<T>::value() const
{
  return value_;
}

template <typename T>
T& Argument<T>::value_mut()
{
  return value_;
}

template <typename T>
template <typename F>
Argument<T>& Argument<T>::action(F&& action) &
{
  static_assert(std::is_invocable_r_v<T, F, std::string_view, Argument<T>*>);
  argparse_argument().action(
    [f = std::forward<F>(action), this](std::string_view value) { return f(value, this); });
  return *this;
}

template <typename T>
bool Argument<T>::was_set() const
{
  return parser_->is_used(flag());
}

// ------------------------------------------------------------------------------------------

template <typename T>
std::ostream& operator<<(std::ostream& os, const Argument<T>& arg)
{
  const auto prev_flags = os.setf(std::ios_base::boolalpha);

  try {
    os << "Argument(flag: " << arg.flag() << ", value: " << arg.value() << ")";
  } catch (...) {
    os.flags(prev_flags);
    throw;
  }
  os.flags(prev_flags);
  return os;
}

}  // namespace legate::detail
