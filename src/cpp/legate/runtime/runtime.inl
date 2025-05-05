/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/runtime.h>

#include <type_traits>

namespace legate {

template <typename T>
void Runtime::register_shutdown_callback(T&& callback)
{
  static_assert(std::is_nothrow_invocable_v<T>);
  register_shutdown_callback_(std::forward<T>(callback));
}

inline Runtime::Runtime(detail::Runtime& runtime) : impl_{&runtime} {}

inline detail::Runtime* Runtime::impl() { return impl_; }

inline const detail::Runtime* Runtime::impl() const { return impl_; }

template <typename T>
void register_shutdown_callback(T&& callback)
{
  Runtime::get_runtime()->register_shutdown_callback(std::forward<T>(callback));
}

}  // namespace legate
