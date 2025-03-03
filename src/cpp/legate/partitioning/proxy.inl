/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/proxy.h>

namespace legate {

constexpr bool ProxyArrayArgument::operator==(const ProxyArrayArgument& rhs) const noexcept
{
  return kind == rhs.kind && index == rhs.index;
}

constexpr bool ProxyArrayArgument::operator!=(const ProxyArrayArgument& rhs) const noexcept
{
  return !(*this == rhs);
}

// ==========================================================================================

namespace proxy_detail {

template <typename T, ProxyArrayArgument::Kind KIND>
constexpr bool TaskArgsBase<T, KIND>::operator==(const TaskArgsBase&) const noexcept
{
  return true;
}

template <typename T, ProxyArrayArgument::Kind KIND>
constexpr bool TaskArgsBase<T, KIND>::operator!=(const TaskArgsBase& rhs) const noexcept
{
  return !(*this == rhs);
}

template <typename T, ProxyArrayArgument::Kind KIND>
constexpr ProxyArrayArgument TaskArgsBase<T, KIND>::operator[](std::uint32_t index) const noexcept
{
  return {KIND, index};
}

}  // namespace proxy_detail

// ==========================================================================================

inline const SharedPtr<detail::ProxyConstraint>& ProxyConstraint::impl() const { return impl_; }

}  // namespace legate
