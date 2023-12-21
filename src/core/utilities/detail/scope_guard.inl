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

#include "core/utilities/detail/scope_guard.h"

#include <utility>

namespace legate::detail {

template <typename T>
ScopeGuard<T>::ScopeGuard(T&& fn) : fn_{std::move(fn)}
{
}

template <typename T>
ScopeGuard<T>::~ScopeGuard()
{
  fn_();
}

// ==========================================================================================

template <typename T>
ScopeGuard<T> make_scope_guard(T&& fn)
{
  return ScopeGuard<T>{std::forward<T>(fn)};
}

}  // namespace legate::detail
