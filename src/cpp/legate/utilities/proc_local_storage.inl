/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/proc_local_storage.h>

#include <cstddef>
#include <typeinfo>

namespace legate {

namespace detail {

[[nodiscard]] std::size_t processor_id();

[[noreturn]] void throw_invalid_proc_local_storage_access(const std::type_info&);

}  // namespace detail

template <typename T>
bool ProcLocalStorage<T>::has_value() const noexcept
{
  return storage_[detail::processor_id()].has_value();
}

template <typename T>
template <typename... Args>
typename ProcLocalStorage<T>::value_type& ProcLocalStorage<T>::emplace(Args&&... args) noexcept(
  std::is_nothrow_constructible_v<value_type, Args...>)
{
  return storage_[detail::processor_id()].emplace(std::forward<Args>(args)...);
}

template <typename T>
constexpr T& ProcLocalStorage<T>::get()
{
  auto& entry = storage_[detail::processor_id()];

  if (!entry.has_value()) {
    detail::throw_invalid_proc_local_storage_access(typeid(T));
  }
  return *entry;
}

template <typename T>
constexpr const T& ProcLocalStorage<T>::get() const
{
  auto& entry = storage_[detail::processor_id()];

  if (!entry.has_value()) {
    detail::throw_invalid_proc_local_storage_access(typeid(T));
  }
  return *entry;
}

}  // namespace legate
