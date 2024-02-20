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

#include "core/utilities/scope_guard.h"

#include <exception>

namespace legate {

template <typename F>
bool& ScopeGuard<F>::enabled_()
{
  return pair_.second();
}

template <typename F>
bool ScopeGuard<F>::enabled_() const
{
  return pair_.second();
}

template <typename F>
typename ScopeGuard<F>::value_type& ScopeGuard<F>::func_()
{
  return pair_.first();
}

template <typename F>
const typename ScopeGuard<F>::value_type& ScopeGuard<F>::func_() const
{
  return pair_.first();
}

// ==========================================================================================

template <typename F>
ScopeGuard<F>::ScopeGuard(value_type&& fn, bool enabled) noexcept
  : pair_{std::forward<value_type>(fn), enabled}
{
}

template <typename F>
ScopeGuard<F>::ScopeGuard(ScopeGuard&& other) noexcept
  : ScopeGuard{std::move(other.func_()), std::exchange(other.enabled_(), false)}
{
}

template <typename F>
ScopeGuard<F>::~ScopeGuard() noexcept
{
  if (enabled()) {
    // Cast to void to explicitly silence any warnings about discarded return-values.
    static_cast<void>(func_()());
  }
}

template <typename F>
ScopeGuard<F>& ScopeGuard<F>::operator=(ScopeGuard&& other) noexcept
{
  if (this != &other) {
    func_()    = std::move(other.func_());
    enabled_() = std::exchange(other.enabled_(), false);
  }
  return *this;
}

template <typename F>
bool ScopeGuard<F>::enabled() const
{
  return enabled_();
}

template <typename F>
void ScopeGuard<F>::disable()
{
  enabled_() = false;
}

template <typename F>
void ScopeGuard<F>::enable()
{
  enabled_() = true;
}

// ==========================================================================================

template <typename F>
ScopeGuard<F> make_scope_guard(F&& fn) noexcept
{
  return ScopeGuard<F>{std::forward<F>(fn)};
}

// ==========================================================================================

template <typename F>
int ScopeFail<F>::exn_count_() const
{
  return exn_cnt_;
}

// ==========================================================================================

template <typename F>
ScopeFail<F>::ScopeFail(value_type&& fn) noexcept
  : guard_{std::forward<value_type>(fn), false}, exn_cnt_{std::uncaught_exceptions()}
{
}

template <typename F>
ScopeFail<F>::~ScopeFail() noexcept
{
  if (std::uncaught_exceptions() != exn_count_()) {
    guard_.enable();
  }
}

// ==========================================================================================

template <typename F>
ScopeFail<F> make_scope_fail(F&& fn) noexcept
{
  return ScopeFail<F>{std::forward<F>(fn)};
}

}  // namespace legate
