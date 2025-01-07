/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <legate/utilities/detail/traced_exception.h>

#include <exception>
#include <utility>

namespace legate::detail {

inline const TracedExceptionBase::Impl* TracedExceptionBase::impl() const noexcept
{
  return impl_.get();
}

// ==========================================================================================

template <typename T>
template <typename... U>
TracedException<T>::TracedException(U&&... args)
  : T{std::forward<U>(args)...},
    TracedExceptionBase{std::make_exception_ptr(static_cast<const T&>(*this)), /* skip_frames */ 1}
{
}

template <typename T>
inline const char* TracedException<T>::what() const noexcept
{
  return TracedExceptionBase::traced_what();
}

// ==========================================================================================

template <typename... T>
TracedException<TracedExceptionBase>::TracedException(T&&...)
{
  // This should really be static_assert(false), but we can't do that.
  static_assert(sizeof...(T) != sizeof...(T),  // NOLINT(misc-redundant-expression)
                "TracedException cannot wrap its base class");
}

// ==========================================================================================

template <typename... T>
TracedException<std::bad_alloc>::TracedException(T&&...)
{
  // This should really be static_assert(false), but we can't do that.
  static_assert(
    sizeof...(T) != sizeof...(T),  // NOLINT(misc-redundant-expression)
    "TracedException allocates on construction, so cannot wrap a std::bad_alloc with it");
}

}  // namespace legate::detail
