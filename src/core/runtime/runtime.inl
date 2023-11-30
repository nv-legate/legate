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

#include "core/runtime/runtime.h"

namespace legate {

template <typename T>
void Runtime::register_shutdown_callback(T&& callback)
{
  static_assert(std::is_nothrow_invocable_v<T>);
  register_shutdown_callback_(std::forward<T>(callback));
}

inline Runtime::Runtime(detail::Runtime* runtime) : impl_{runtime} {}

inline detail::Runtime* Runtime::impl() { return impl_; }

template <typename T>
void register_shutdown_callback(T&& callback)
{
  Runtime::get_runtime()->register_shutdown_callback(std::forward<T>(callback));
}

}  // namespace legate
