/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/internal_shared_ptr.h>

#include <legate/utilities/detail/traced_exception.h>

#include <string>
#include <utility>

namespace legate {

BadInternalWeakPtr::BadInternalWeakPtr(std::string what) noexcept : what_{std::move(what)} {}

const char* BadInternalWeakPtr::what() const noexcept { return what_.c_str(); }

namespace detail {

void throw_bad_internal_weak_ptr()
{
  throw detail::TracedException<BadInternalWeakPtr>{
    "Trying to construct an InternalSharedPtr from an empty InternalWeakPtr"};
}

}  // namespace detail

}  // namespace legate
